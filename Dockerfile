# This Dockerfile is designed to deliver vllm-detector-adapter
# in a tiny image only containing this app + any additional required
# packages. (At the time of writing, there are no additional packages required)
# NOTE: The image generated out of this file should only be used in conjunction with a
# compatible vllm image, otherwise it will throw error.

ARG BASE_UBI_IMAGE_TAG=9.6
ARG PYTHON_VERSION=3.12

### Build layer
FROM quay.io/vllm/vllm-cuda:0.9.2.2 as build

ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}

USER root

COPY scripts/targeted_pip_install.py /app/

COPY vllm_detector_adapter /app/vllm-detector-adapter/vllm_detector_adapter
COPY pyproject.toml /app/vllm-detector-adapter

WORKDIR /app

RUN python${PYTHON_VERSION} targeted_pip_install.py -p /app/vllm-detector-adapter/ -t /app/target_packages

### Build layer with tgis-adapter
FROM build as build_tgis_adapter

WORKDIR /app/vllm-detector-adapter

# Because tgis-adapter is an "extra" that we need to specify, we can't provide installation directory path
# directly. So we need to use the `.` operator and to be able to do that, we need to run install script
# via full path.
RUN python${PYTHON_VERSION} /app/targeted_pip_install.py -p .["vllm-tgis-adapter"] -t /app/target_packages

### Release Layer (only vllm-detector-adapter)
FROM registry.access.redhat.com/ubi9/ubi-minimal:${BASE_UBI_IMAGE_TAG} as release

COPY --from=build --chown=1001 /app/target_packages /app/target_packages

# Image hardening
RUN --mount=type=bind,source=scripts/image_hardening,target=scripts \
    sh scripts/installRemediationTools.sh && \
    sh scripts/remediation-script.sh && \
    sh scripts/removeRemediationTools.sh

USER 1001

HEALTHCHECK NONE

ENV SHARED_PACKAGE_PATH="/shared_packages/app/"

# The entrypoint for this image is designed to follow its usage, i.e
# to be used along with vllm image. Therefore, in this image, we
# only copy the package(s) to a shared package path
# NOTE: Along with this, one may need to adjust PYTHONPATH to call
# this package, depending on setup.
# Example: PYTHONPATH='${SHARED_PACKAGE_PATH}:${PYTHONPATH}'
ENTRYPOINT ["/bin/bash", "-c", "cp -r /app/target_packages/*  ${SHARED_PACKAGE_PATH}" ]

### Release Layer (with vllm-tgis-adapter)
FROM release as release_tgis_adapter

COPY --from=build_tgis_adapter --chown=1001 /app/target_packages /app/target_packages

# Image hardening
RUN --mount=type=bind,source=scripts/image_hardening,target=scripts \
    sh scripts/installRemediationTools.sh && \
    sh scripts/remediation-script.sh && \
    sh scripts/removeRemediationTools.sh

USER 1001

HEALTHCHECK NONE
