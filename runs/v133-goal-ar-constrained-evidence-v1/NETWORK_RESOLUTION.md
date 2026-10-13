# SSH connection restored

On the user-authorized recovery attempt, the configured command failed DNS resolution in the default command sandbox and immediately succeeded outside that sandbox, using the same user, hostname and port. No DNS/hosts/SSH settings were changed. The cause of the earlier connection barrier was the command sandbox's network restrictions, not evidence that the AutoDL instance was off. The previous startup records are retained.

Remote hostname: autodl-container-ef014e83a6-9085728e. GPU: NVIDIA GeForce RTX 4090 D. Python: 3.12.3. Actual Goal Z and AM adapter hashes match the frozen identities. Continue all SSH and file-transfer operations through the authorized outside-sandbox command path.
