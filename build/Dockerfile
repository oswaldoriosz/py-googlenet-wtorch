# File: builder/Dockerfile
FROM registry.redhat.io/ubi9/python-312
USER root
RUN dnf install -y gcc gcc-c++ cmake make python3-devel
RUN pip install --upgrade pip
USER 1001
CMD ["/bin/bash"]