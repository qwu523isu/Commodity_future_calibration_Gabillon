# Delivery and deployment plan

## Implemented delivery workflow

On a push to `main`, CI runs the Python tests, builds the image, exercises the actual container and checks persisted results after container replacement. Only then does it publish **that same tested image** to:

```text
ghcr.io/qwu523isu/commodity_future_calibration_gabillon:sha-<full-commit-SHA>
```

PRs and manual CI dispatches run tests without publishing. Authentication uses the job-scoped `GITHUB_TOKEN` with `packages: write`; no personal token is stored in the repository. The Actions job summary records the source commit, image tag and immutable registry digest. The first publication cannot be verified until this workflow is merged and a main-branch push runs it.

GitHub packages are private by default on first publication. A public source repository does not automatically make the package publicly pullable. For a simple demo, review the package visibility in GitHub Packages and choose public if desired. Otherwise authenticate the deployment host using an appropriate package-read credential stored outside source control. Reference: [GitHub Container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

This is continuous delivery of an image, not automatic server deployment. No cloud resources have been provisioned.

## Suggested initial cloud target

For the existing single-container/file-storage architecture, a small Linux VM is the least disruptive target. AWS Lightsail is one option: Ubuntu, Docker and a named volume can run the existing image without changing the storage layer. Start with a small demonstration workload, then measure memory and latency before selecting a larger plan. A single VM is not highly available; persistent volumes still require backups.

Cloud Run is another option, but local container files are not durable storage. Supporting reliable cross-instance result retrieval would require external storage and additional application changes. It is therefore a later option if managed scaling becomes the priority.

Platform, region, spending limit and account access must be chosen before provisioning. No cloud credentials should be committed or pasted into chat.

## Manual demo deployment after choosing a host

On a Linux machine with Docker installed, replace the digest below with the **Immutable image** value from the successful publication's Actions summary:

```sh
IMAGE='ghcr.io/qwu523isu/commodity_future_calibration_gabillon@sha256:<digest>'
docker pull "$IMAGE"
docker run -d --name risk-api --restart unless-stopped \
  -p 127.0.0.1:8000:8000 -v risk-api-results:/data "$IMAGE"
curl --fail http://127.0.0.1:8000/health
```

Keep port 8000 local to the VM. For an interview demonstration, use an SSH tunnel from your laptop:

```sh
ssh -L 8000:127.0.0.1:8000 <user>@<server>
```

Then submit the configuration to `http://127.0.0.1:8000/runs` from the laptop. The current API is synchronous, unauthenticated and intended for demonstration; internet-facing hosting needs an appropriate serving/authentication layer.

## Update and rollback

Record the currently deployed digest before changing versions. Pull a new digest before stopping the existing container, then replace the container using the same named volume. Verify health and an actual risk calculation. If verification fails, recreate the container using the previous digest and the same volume. This simple replacement has downtime and is not yet automated. Do not remove the named volume during rollback.

Future server CD should implement this procedure as a controlled deployment job with environment-specific access, health checks and rollback. The current repository only publishes tested images; it does not connect to or alter a server.
