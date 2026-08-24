# The cases below drive the conversion from the resources a container declares
# to the requirements a Pod carries, which is where the requests/limits split
# becomes observable without a live Kubernetes cluster.
# ruff: noqa: SLF001

import logging

import kubernetes.client
import pytest

from gpustack_runtime.deployer.__types__ import (
    Container,
    ContainerProfileEnum,
    ContainerResources,
    Deployer,
)
from gpustack_runtime.deployer.kuberentes import (
    KubernetesDeployer,
    KubernetesWorkloadPlan,
    _container_resource_requirements,
    _is_overcommittable_resource,
)


def _resources(**kwargs) -> ContainerResources:
    resources = ContainerResources()
    resources.update(kwargs)
    return resources


@pytest.mark.parametrize(
    "name, resource_key, expected",
    [
        ("cpu", "cpu", True),
        ("memory", "memory", True),
        ("ephemeral storage", "ephemeral-storage", True),
        ("hugepages", "hugepages-2Mi", True),
        ("device plugin resource", "nvidia.com/gpu", False),
        ("sliced device plugin resource", "huawei.com/npu.sliced", False),
        ("gpustack automap key", "gpustack.ai/devices", False),
    ],
)
def test_is_overcommittable_resource(name, resource_key, expected):
    actual = _is_overcommittable_resource(resource_key)
    assert actual == expected, f"case {name} expected {expected}, but got {actual}"


def test_resolve_resources_defaults_the_limits_to_the_requests():
    # A container declaring one side only requests and is limited to the same
    # resources, which is what every container has got so far.
    requests = _resources(cpu=2, memory="4Gi")
    container = Container(name="run", image="busybox:1.37", resources=requests)

    assert container.resolve_resources() == (requests, requests)


def test_resolve_resources_defaults_the_requests_to_the_limits():
    limits = _resources(cpu=2, memory="4Gi")
    container = Container(name="run", image="busybox:1.37", resources_limits=limits)

    assert container.resolve_resources() == (limits, limits)


def test_resolve_resources_keeps_a_declared_split_apart():
    requests = _resources(cpu=2)
    limits = _resources(cpu=4)
    container = Container(
        name="run",
        image="busybox:1.37",
        resources=requests,
        resources_limits=limits,
    )

    assert container.resolve_resources() == (requests, limits)


def test_resolve_resources_of_a_container_declaring_none():
    container = Container(name="run", image="busybox:1.37")

    assert container.resolve_resources() == (None, None)


def test_container_resource_requirements_without_limits_stays_guaranteed():
    requirements = _container_resource_requirements(
        "run",
        {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"},
        None,
    )

    assert requirements.requests == requirements.limits
    assert requirements.requests == {
        "cpu": "2",
        "memory": "4Gi",
        "nvidia.com/gpu": "1",
    }


def test_container_resource_requirements_mirroring_the_requests_stays_guaranteed():
    # A container declaring one side only hands the same mapping in twice,
    # which must convert to exactly what it always has.
    requests = {"cpu": "2", "memory": "4Gi"}

    requirements = _container_resource_requirements(
        "run",
        requests,
        _resources(cpu=2, memory="4Gi"),
    )

    assert requirements.requests == requests
    assert requirements.limits == requests


def test_container_resource_requirements_with_limits_turns_burstable():
    requirements = _container_resource_requirements(
        "run",
        {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"},
        _resources(cpu=4, memory="8Gi"),
    )

    assert requirements.requests == {
        "cpu": "2",
        "memory": "4Gi",
        "nvidia.com/gpu": "1",
    }
    # The device request is limited to what it requests whatever the limits
    # say, as Kubernetes rejects an extended resource declaring them apart.
    assert requirements.limits == {
        "cpu": "4",
        "memory": "8Gi",
        "nvidia.com/gpu": "1",
    }


def test_container_resource_requirements_caps_an_extended_resource(caplog):
    with caplog.at_level(logging.WARNING):
        requirements = _container_resource_requirements(
            "run",
            {"nvidia.com/gpu": "1"},
            _resources(**{"nvidia.com/gpu": 2}),
        )

    assert requirements.requests == {"nvidia.com/gpu": "1"}
    assert requirements.limits == {"nvidia.com/gpu": "1"}
    assert "nvidia.com/gpu" in caplog.text


def test_container_resource_requirements_ignores_an_unrequested_limit():
    # A mapped device request never reaches the Pod as a resource under the env
    # injection policy, so a limit naming it has nothing to limit.
    requirements = _container_resource_requirements(
        "run",
        {"cpu": "2"},
        _resources(cpu=2, **{"gpustack.ai/devices": "0,1"}),
    )

    assert requirements.requests == {"cpu": "2"}
    assert requirements.limits == {"cpu": "2"}


def test_container_resource_requirements_of_a_container_declaring_none():
    requirements = _container_resource_requirements("run", {}, None)

    assert requirements.requests is None
    assert requirements.limits is None


def _create_pod(monkeypatch, container: Container) -> kubernetes.client.V1Pod:
    monkeypatch.setattr(
        "gpustack_runtime.deployer.kuberentes.get_resource_injection_policy",
        lambda *_args: "env",
    )
    monkeypatch.setattr(
        "gpustack_runtime.deployer.kuberentes._resolve_runtime_class_name",
        lambda *_args: None,
    )

    class _FakeCoreV1Api:
        def __init__(self, client=None):
            pass

        def read_namespaced_pod(self, name, namespace):
            raise kubernetes.client.exceptions.ApiException(status=404)

        def create_namespaced_pod(self, namespace, body):
            return body

    monkeypatch.setattr(kubernetes.client, "CoreV1Api", _FakeCoreV1Api)

    deployer = object.__new__(KubernetesDeployer)
    Deployer.__init__(deployer, "test")
    deployer._materials = {}
    deployer._client = None
    deployer._node_name = None
    deployer._image_pull_secrets = {}
    deployer._mutate_create_pod = lambda pod: pod

    workload = KubernetesWorkloadPlan(
        name="test",
        namespace="gpustack-default",
        containers=[container],
    )
    workload.validate_and_default()

    return deployer._create_pod(workload, {})


def test_create_pod_carries_the_requests_and_the_limits_apart(monkeypatch):
    pod = _create_pod(
        monkeypatch,
        Container(
            name="run",
            image="busybox:1.37",
            profile=ContainerProfileEnum.RUN,
            resources=_resources(cpu=2, memory="4Gi"),
            resources_limits=_resources(cpu=4, memory="8Gi"),
        ),
    )

    resources = pod.spec.containers[0].resources
    assert resources.requests == {"cpu": "2", "memory": "4Gi"}
    assert resources.limits == {"cpu": "4", "memory": "8Gi"}


def test_create_pod_of_a_container_declaring_limits_only(monkeypatch):
    # Kubernetes defaults a request to the limit of the same resource, so a
    # container declaring limits alone requests exactly them.
    pod = _create_pod(
        monkeypatch,
        Container(
            name="run",
            image="busybox:1.37",
            profile=ContainerProfileEnum.RUN,
            resources_limits=_resources(cpu=2, memory="4Gi"),
        ),
    )

    resources = pod.spec.containers[0].resources
    assert resources.requests == {"cpu": "2", "memory": "4Gi"}
    assert resources.limits == {"cpu": "2", "memory": "4Gi"}


def test_create_pod_of_a_container_declaring_requests_only(monkeypatch):
    pod = _create_pod(
        monkeypatch,
        Container(
            name="run",
            image="busybox:1.37",
            profile=ContainerProfileEnum.RUN,
            resources=_resources(cpu=2, memory="4Gi"),
        ),
    )

    resources = pod.spec.containers[0].resources
    assert resources.requests == {"cpu": "2", "memory": "4Gi"}
    assert resources.limits == {"cpu": "2", "memory": "4Gi"}
