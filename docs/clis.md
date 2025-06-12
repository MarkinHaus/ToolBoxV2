Of course. Here are three rich helper guides and usage examples for your enhanced CLI applications, formatted in Markdown.

---

## 🚀 P2P Tunnel Manager (`tmc_p2p_cli.py`) - User Guide

This utility provides a robust command-line interface to manage instances of the P2P tunneling application. It creates isolated environments for each peer and relay, making it easy to configure, run, and debug complex network setups.

### Quickstart

1.  **Build the application:**
    ```bash
    tb p2p build
    ```

2.  **Start a Relay Server:**
    ```bash
    tb p2p start-relay main-relay --password "a-secure-password"
    ```

3.  **Start Peers:**
    *   **Provider** (exposes a local service):
        ```bash
        tb p2p start-peer api-provider --peer-id service-A \
          --relay-addr 127.0.0.1:9000 --relay-pass "a-secure-password" \
          --forward 127.0.0.1:3000
        ```
    *   **Consumer** (accesses the provider's service):
        ```bash
        tb p2p start-peer api-consumer --target service-A \
          --relay-addr 127.0.0.1:9000 --relay-pass "a-secure-password" \
          --listen 127.0.0.1:8000
        ```

4.  **Check Status & Logs:**
    ```bash
    tb p2p status
    tb p2p logs api-provider
    ```

5.  **Stop Instances:**
    ```bash
    # Stop one instance
    tb p2p stop api-consumer

    # Stop all instances
    tb p2p stop
    ```

## 🚀 Blob DB Cluster Manager (`db_cli.py`) - User Guide

This CLI is a powerful tool for managing a distributed cluster of `r_blob_db` instances. It handles configuration, state, and provides commands for starting, stopping, and health-checking the entire cluster or individual nodes.

### Quickstart

1.  **Build the application:**
    ```bash
    tb db build
    ```

2.  **Initialize the Cluster:** The first time you run a command, a `cluster_config.json` is created with a default two-node setup. You can customize this file.

3.  **Start the Cluster:**
    ```bash
    # Starts all instances defined in cluster_config.json
    tb db start
    ```

4.  **Check Status & Health:**
    ```bash
    tb db status
    tb db health
    ```

5.  **Stop the Cluster:**
    ```bash
    # Stop a single instance
    tb db stop --instance-id instance-01

    # Stop all instances
    tb db stop
    ```

### Example 2: Performing a Rolling Update on the Live Cluster

**Scenario:** You've developed `v1.1.0` of `r_blob_db` and need to update your running `v1.0.0` cluster without any downtime. The rolling update process updates one node at a time, ensuring the cluster remains available.

1.  **Check Current Cluster Health:**
    Before starting, ensure all nodes are healthy.
    ```bash
    tb db health
    ```
    *You should see all instances report `✅ OK`.*

2.  **Build the New Version:**
    Compile the new version of your application. The manager will automatically find the new binary.
    ```bash
    # Assuming your code is updated to v1.1.0
    tb db build
    ```

3.  **Initiate the Rolling Update:**
    Execute the `update` command, specifying the new version string.
    ```bash
    tb db update --version "v1.1.0"
    ```

4.  **Monitor the Process:**
    The CLI will provide detailed, real-time feedback:
    ```text
    --- Starting Rolling Update to Version v1.1.0 ---

    [1/2] Updating instance 'instance-01'...
    ⏹️  Instance 'instance-01' stopped.
    🚀 Starting instance 'instance-01' on port 3001...
    ✅ Instance 'instance-01' started successfully. (PID: 12346)
    ...
    ⧖ Waiting for 'instance-01' to become healthy...
    ✅ Instance 'instance-01' is healthy with new version.

    [2/2] Updating instance 'instance-02'...
    ...
    --- Rolling Update Complete ---
    ```

5.  **Verify the Update:**
    Run the health check again. All instances should now report `OK` and show `server_version: v1.1.0`.
    ```bash
    tb db health
    ```

---

## 🚀 API Server Manager (`api_manager.py`) - User Guide

This manager is designed for high-availability web services. Its standout feature is the ability to perform **zero-downtime updates on POSIX systems (Linux/macOS)** by passing the active network socket from the old process to the new one, ensuring no client requests are dropped during the update.

### Quickstart

1.  **Build the application:**
    ```bash
    # Assuming the CLI entrypoint is mapped to `tb`
    tb api build
    ```

2.  **Start the Server:**
    *   **On Linux/macOS (with Zero-Downtime enabled):**
        ```bash
        tb api start --posix-zdt
        ```
    *   **On Windows (uses graceful restart):**
        ```bash
        tb api start
        ```

3.  **Check Status:**
    ```bash
    tb api status
    ```

4.  **Update the Server:**
    ```bash
    # First, build the new version
    tb api build

    # Then, run the update
    tb api update --version "v1.2.0" --posix-zdt
    ```

5.  **Stop the Server:**
    ```bash
    tb api stop
    ```

### Example 3: Zero-Downtime Deployment on a Linux Server

**Scenario:** Your API server is handling live traffic. You need to deploy a critical security patch (`v1.0.1`) without interrupting any ongoing client connections.

1.  **Check Initial State:**
    Ensure the server is running correctly. The `--posix-zdt` flag confirms that the manager is aware of the socket file descriptor.
    ```bash
    tb api status --posix-zdt
    ```
    *Output:*
    ```text
    --- Server Status ---
      ✅ RUNNING
        PID:        11223
        Version:    v1.0.0
        Executable: /path/to/project/src-core/simple-core-server
        Listening FD: 4 (POSIX ZDT Active)
    ```

2.  **Build the New Version:**
    Compile the patched version of the code.
    ```bash
    tb api build
    ```

3.  **Execute the Zero-Downtime Update:**
    Run the `update` command with the `--posix-zdt` flag.
    ```bash
    tb api update --version "v1.0.1" --posix-zdt
    ```

4.  **Observe the Magic:**
    The manager performs the following sequence seamlessly:
    *   It finds the persistent socket file descriptor (`FD: 4`).
    *   It starts the **new** server process (`v1.0.1`), passing it ownership of the active socket. The new server begins accepting new connections on the same port immediately.
    *   Once the new server is running, the manager sends a `SIGTERM` signal to the **old** process (`v1.0.0`).
    *   The old process stops accepting new connections but finishes handling any in-flight requests before shutting down.
    *   The state file is updated with the new PID and version.

    *Terminal Output:*
    ```text
    --- [POSIX] Starting Zero-Downtime Update to v1.0.1 ---
    ✅ New server started (PID: 11255).
    ⏹️  Process 11223 stopped.
    --- Update Complete. New PID: 11255 ---
    ```

5.  **Final Verification:**
    Check the status again. The server is still `RUNNING`, but now with the new PID and version. No client would have noticed the switch.
    ```bash
    tb api status --posix-zdt
    ```

and wit h the same focus new the last ui update + use real Posix by global fag: import contextlib

