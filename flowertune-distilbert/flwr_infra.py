import subprocess
import time
import os
import sys
import logging
import tomllib

# (count, cpus) tuples defining client groups.
DEFAULT_CLIENT_GROUPS = [
    (1, 8),
    (1, 4),
    (1, 2),
    (1, 1),
]

# Configuration
SIF_FILE = "flwr.sif"
SUPERLINK_PORTS = {
    "serverappio": 54001,
    "fleet": 54002,
    "control": 54003
}
SUPERNODE_PORT_START = 54100  # Starting port for Supernodes (clientappio API)
CLIENT_GROUPS_CONFIG = "client_groups.toml"

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_client_groups(config_path: str):
    if not os.path.exists(config_path):
        logger.info(
            "Client group config '%s' not found, using defaults",
            config_path,
        )
        return DEFAULT_CLIENT_GROUPS

    try:
        with open(config_path, "rb") as config_file:
            data = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.error("Failed to read '%s': %s", config_path, exc)
        logger.info("Falling back to default client groups")
        return DEFAULT_CLIENT_GROUPS

    raw_groups = data.get("client_groups", [])
    parsed_groups = []

    for index, group in enumerate(raw_groups):
        count = group.get("count")
        cpus = group.get("cpus")

        if not isinstance(count, int) or count < 1:
            logger.error("Invalid count in client_groups[%s]: %r", index, count)
            continue

        if not isinstance(cpus, int) or cpus < 1:
            logger.error("Invalid cpus in client_groups[%s]: %r", index, cpus)
            continue

        parsed_groups.append((count, cpus))

    if not parsed_groups:
        logger.error("No valid client_groups found in '%s'", config_path)
        logger.info("Falling back to default client groups")
        return DEFAULT_CLIENT_GROUPS

    logger.info("Loaded %s client group entries from '%s'", len(parsed_groups), config_path)
    logger.debug("Resolved client groups: %s", parsed_groups)
    return parsed_groups


def start_instances():
    """Spawns the Superlink and the heterogeneous Supernodes."""
    if not os.path.exists(SIF_FILE):
        logger.error("%s not found. Run 'apptainer pull' first.", SIF_FILE)
        return

    # 1. Start Superlink
    logger.info("Launching Superlink (logs: %s/superlink.log)", LOG_DIR)
    logger.debug("Starting instance: superlink")
    subprocess.run(["apptainer", "instance", "start",
                   SIF_FILE, "superlink"], check=False)

    sl_log = open(f"{LOG_DIR}/superlink.log", "w")
    subprocess.Popen([
        "apptainer", "exec", "instance://superlink",
        "flower-superlink", "--insecure", "--isolation", "subprocess",
        "--serverappio-api-address", f"0.0.0.0:{SUPERLINK_PORTS['serverappio']}",
        "--fleet-api-address", f"0.0.0.0:{SUPERLINK_PORTS['fleet']}",
        "--control-api-address", f"0.0.0.0:{SUPERLINK_PORTS['control']}"
    ], stdout=sl_log, stderr=sl_log, start_new_session=True)
    logger.debug("Superlink process started")

    # Wait for Link to bind ports 9091-9093
    time.sleep(5)

    # 2. Start Heterogeneous Supernodes
    client_groups = load_client_groups(CLIENT_GROUPS_CONFIG)
    node_id = 0
    total_nodes = sum(count for count, _ in client_groups)
    num_cores = os.cpu_count()
    current_core_ptr = 0  # Tracks the next available core for allocation
    if not num_cores:
        num_cores = 1
        logger.debug("os.cpu_count() returned None. Falling back to 1 core.")

    logger.info("Spawning %s heterogeneous supernodes", total_nodes)

    for count, cpus in client_groups:
        cores_to_allocate = max(1, int(cpus))
        for _ in range(count):
            instance_name = f"node-{node_id}"
            port = SUPERNODE_PORT_START + node_id

            # Create a list of cores for this specific node
            assigned_cores = []
            for i in range(cores_to_allocate):
                assigned_cores.append(str((current_core_ptr + i) % num_cores))
            core_mask = ",".join(assigned_cores)  # e.g., "0,1,2,3"
            current_core_ptr = (current_core_ptr +
                                cores_to_allocate) % num_cores
            logger.debug(
                "Node %s config: cpus=%s, core_mask=%s, port=%s",
                node_id,
                cpus,
                core_mask,
                port,
            )

            # Start the background instance shell
            subprocess.run(["apptainer", "instance", "start",
                           SIF_FILE, instance_name], check=False)

            node_log = open(f"{LOG_DIR}/{instance_name}.log", "w")
            exec_cmd = [
                "taskset", "-c", core_mask,
                "apptainer", "exec",
                "--env", f"OMP_NUM_THREADS={int(cpus)}",
                "--env", f"MKL_NUM_THREADS={int(cpus)}",
                f"instance://{instance_name}",
                "flower-supernode", "--insecure",
                "--superlink", f"127.0.0.1:{SUPERLINK_PORTS['fleet']}",
                "--clientappio-api-address", f"0.0.0.0:{port}",
                "--isolation", "subprocess",
                "--node-config", f"partition-id={node_id} num-partitions={total_nodes}"
            ]

            subprocess.Popen(exec_cmd, stdout=node_log,
                             stderr=node_log, start_new_session=True)
            logger.info("Started %s on port %s (cores: %s)",
                        instance_name, port, core_mask)
            node_id += 1

    logger.info("Current Apptainer instance status:")
    subprocess.run(["apptainer", "instance", "list"], check=False)


def stop_instances():
    """Stops all running Apptainer instances on this host."""
    logger.info("Stopping all Flower instances")
    # Stopping all is the safest way to ensure no orphaned nodes remain
    result = subprocess.run(
        ["apptainer", "instance", "stop", "--all"], check=False)
    if result.returncode == 0:
        logger.info("Successfully stopped all instances")
    else:
        logger.error("No active instances found or error during shutdown")


if __name__ == "__main__":
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))

    if len(sys.argv) < 2:
        logger.error("Usage: python run_fl.py [start|stop]")
        sys.exit(1)

    action = sys.argv[1].lower()
    if action == "start":
        start_instances()
    elif action == "stop":
        stop_instances()
    else:
        logger.error("Unknown action: %s. Use 'start' or 'stop'.", action)
