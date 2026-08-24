# The cases below drive the resolution of the namespace a workload lives in,
# which a worker hosting the workloads of several organizations can no longer
# read off its own namespace.
# ruff: noqa: SLF001

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import kubernetes.client
import pytest

from gpustack_runtime import envs
from gpustack_runtime.deployer.__types__ import (
    Container,
    ContainerProfileEnum,
    Deployer,
)
from gpustack_runtime.deployer.kuberentes import (
    _LABEL_WORKLOAD,
    KubernetesDeployer,
    KubernetesWorkloadPlan,
    _default_workload_namespace,
    _is_workload_namespace,
)

_AGENT_NAMESPACE = "gpustack-system"
"""
Namespace the worker agent itself runs in, which is also the one the deployer
falls back to and the one Kueue cannot admit workloads in.
"""


@pytest.fixture(autouse=True)
def _agent_namespace(monkeypatch):
    monkeypatch.setattr(
        envs,
        "GPUSTACK_RUNTIME_KUBERNETES_NAMESPACE",
        _AGENT_NAMESPACE,
    )


def _pod(
    namespace: str,
    name: str = "test",
    node_name: str = "node-0",
) -> kubernetes.client.V1Pod:
    return kubernetes.client.V1Pod(
        metadata=kubernetes.client.V1ObjectMeta(
            name=f"gpustack-{name}",
            namespace=namespace,
            labels={_LABEL_WORKLOAD: name},
            creation_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        spec=kubernetes.client.V1PodSpec(
            containers=[kubernetes.client.V1Container(name="run")],
            node_name=node_name,
        ),
        status=kubernetes.client.V1PodStatus(
            phase="Running",
            container_statuses=[
                kubernetes.client.V1ContainerStatus(
                    name="run",
                    image="busybox:1.37",
                    image_id="",
                    ready=True,
                    restart_count=0,
                    state=kubernetes.client.V1ContainerState(),
                ),
            ],
        ),
    )


def _fake_core_api(
    monkeypatch,
    all_namespaces: list[kubernetes.client.V1Pod] | None = None,
    namespaced: dict[str, list[kubernetes.client.V1Pod]] | None = None,
    all_namespaces_status: int | None = None,
):
    """
    Patch CoreV1Api so that listing Pods answers from the given fixtures, and
    record the calls it receives.
    """
    journal = []

    class _FakeCoreV1Api:
        def __init__(self, client=None):
            pass

        def list_pod_for_all_namespaces(self, **kwargs):
            journal.append(("list_pod_for_all_namespaces", kwargs))
            if all_namespaces_status:
                raise kubernetes.client.exceptions.ApiException(
                    status=all_namespaces_status,
                )
            return SimpleNamespace(items=list(all_namespaces or []))

        def list_namespaced_pod(self, namespace, **kwargs):
            journal.append(("list_namespaced_pod", {"namespace": namespace, **kwargs}))
            return SimpleNamespace(items=list((namespaced or {}).get(namespace, [])))

    monkeypatch.setattr(kubernetes.client, "CoreV1Api", _FakeCoreV1Api)
    return journal


def _deployer(node_name: str = "node-0") -> KubernetesDeployer:
    # Bypass the deployer's __init__, which reaches out to a live API server.
    deployer = object.__new__(KubernetesDeployer)
    Deployer.__init__(deployer, "test")
    deployer._materials = {}
    deployer._client = None
    deployer._node_name = node_name
    deployer._image_pull_secrets = {}
    deployer._mutate_create_pod = lambda pod: pod
    deployer.is_supported = lambda: True
    return deployer


def test_default_workload_namespace_reads_the_configured_one():
    assert _default_workload_namespace() == _AGENT_NAMESPACE


@pytest.mark.parametrize(
    "name, namespace, expected",
    [
        ("organization namespace", "gpustack-default", True),
        ("another organization namespace", "gpustack-acme", True),
        ("configured default", _AGENT_NAMESPACE, True),
        ("system namespace", "kube-system", False),
        ("unrelated namespace", "default", False),
        ("no namespace", None, False),
        ("empty namespace", "", False),
    ],
)
def test_is_workload_namespace(name, namespace, expected):
    actual = _is_workload_namespace(namespace)
    assert actual == expected, f"case {name} expected {expected}, but got {actual}"


def test_workload_namespace_prefers_the_declared_one(monkeypatch):
    journal = _fake_core_api(monkeypatch)
    deployer = _deployer()

    assert deployer._workload_namespace(namespace="gpustack-acme") == "gpustack-acme"
    # A declared namespace needs no search at all.
    assert not journal


def test_workload_namespace_searches_the_workload_namespaces(monkeypatch):
    _fake_core_api(monkeypatch, all_namespaces=[_pod("gpustack-acme")])
    deployer = _deployer()

    assert deployer._workload_namespace(name="test") == "gpustack-acme"


def test_workload_namespace_falls_back_to_the_configured_one(monkeypatch):
    _fake_core_api(monkeypatch, all_namespaces=[])
    deployer = _deployer()

    assert deployer._workload_namespace(name="test") == _AGENT_NAMESPACE


def test_workload_namespace_falls_back_when_the_search_is_forbidden(monkeypatch):
    journal = _fake_core_api(monkeypatch, all_namespaces_status=403)
    deployer = _deployer()

    assert deployer._workload_namespace(name="test") == _AGENT_NAMESPACE
    # The cluster-wide read degrades to the only namespace the deployer used to
    # read, so a cluster granting no more permission than before keeps working.
    assert journal[-1][0] == "list_namespaced_pod"
    assert journal[-1][1]["namespace"] == _AGENT_NAMESPACE


def test_workload_namespace_picks_deterministically_on_a_name_collision(
    monkeypatch,
    caplog,
):
    _fake_core_api(
        monkeypatch,
        all_namespaces=[_pod("gpustack-beta"), _pod("gpustack-acme")],
    )
    deployer = _deployer()

    with caplog.at_level(logging.WARNING):
        assert deployer._workload_namespace(name="test") == "gpustack-acme"

    assert "gpustack-beta" in caplog.text


def test_list_workload_pods_reads_a_declared_namespace_only(monkeypatch):
    journal = _fake_core_api(
        monkeypatch,
        namespaced={"gpustack-acme": [_pod("gpustack-acme")]},
    )
    deployer = _deployer()

    k_pods = deployer._list_workload_pods(namespace="gpustack-acme")

    assert [k_pod.metadata.namespace for k_pod in k_pods] == ["gpustack-acme"]
    assert [c[0] for c in journal] == ["list_namespaced_pod"]


def test_list_workload_pods_keeps_the_workload_namespaces_only(monkeypatch):
    _fake_core_api(
        monkeypatch,
        all_namespaces=[
            _pod("gpustack-acme"),
            _pod("kube-system"),
            _pod(_AGENT_NAMESPACE),
        ],
    )
    deployer = _deployer()

    k_pods = deployer._list_workload_pods()

    assert [k_pod.metadata.namespace for k_pod in k_pods] == [
        "gpustack-acme",
        _AGENT_NAMESPACE,
    ]


def test_get_finds_the_workload_of_another_namespace(monkeypatch):
    _fake_core_api(monkeypatch, all_namespaces=[_pod("gpustack-acme")])
    deployer = _deployer()

    status = KubernetesDeployer._get(deployer, name="test")

    assert status is not None
    assert status.namespace == "gpustack-acme"


def test_get_reads_a_declared_namespace_only(monkeypatch):
    journal = _fake_core_api(
        monkeypatch,
        namespaced={"gpustack-acme": [_pod("gpustack-acme")]},
    )
    deployer = _deployer()

    status = KubernetesDeployer._get(deployer, name="test", namespace="gpustack-acme")

    assert status is not None
    assert [c[0] for c in journal] == ["list_namespaced_pod"]


def test_list_sweeps_the_workload_namespaces(monkeypatch):
    _fake_core_api(
        monkeypatch,
        all_namespaces=[
            _pod("gpustack-acme", name="acme"),
            _pod("gpustack-beta", name="beta"),
            _pod("kube-system", name="foreign"),
        ],
    )
    deployer = _deployer()

    statuses = KubernetesDeployer._list(deployer)

    assert [status.namespace for status in statuses] == [
        "gpustack-acme",
        "gpustack-beta",
    ]


def test_delete_removes_from_the_namespace_holding_the_workload(monkeypatch):
    journal = []

    class _FakeCoreV1Api:
        def __init__(self, client=None):
            pass

        def __getattr__(self, name):
            def call(**kwargs):
                journal.append((name, kwargs))

            return call

    monkeypatch.setattr(kubernetes.client, "CoreV1Api", _FakeCoreV1Api)

    dep = SimpleNamespace(
        is_supported=lambda: True,
        get=lambda **_kwargs: SimpleNamespace(name="test", namespace="gpustack-acme"),
        _client=None,
    )

    KubernetesDeployer._delete(dep, name="test")

    # Deleting from the deployer's own namespace instead would leave the Pod
    # running, and with it the devices it holds.
    assert journal
    assert {c[1]["namespace"] for c in journal} == {"gpustack-acme"}


def test_delete_falls_back_to_the_configured_namespace(monkeypatch):
    journal = []

    class _FakeCoreV1Api:
        def __init__(self, client=None):
            pass

        def __getattr__(self, name):
            def call(**kwargs):
                journal.append((name, kwargs))

            return call

    monkeypatch.setattr(kubernetes.client, "CoreV1Api", _FakeCoreV1Api)

    dep = SimpleNamespace(
        is_supported=lambda: True,
        get=lambda **_kwargs: SimpleNamespace(name="test", namespace=None),
        _client=None,
    )

    KubernetesDeployer._delete(dep, name="test")

    assert {c[1]["namespace"] for c in journal} == {_AGENT_NAMESPACE}


def _fake_secret_api(monkeypatch):
    """
    Patch CoreV1Api so that applying a Secret records the namespace it lands in.
    """
    journal = []

    class _FakeCoreV1Api:
        def __init__(self, client=None):
            pass

        def read_namespaced_secret(self, name, namespace):
            raise kubernetes.client.exceptions.ApiException(status=404)

        def create_namespaced_secret(self, namespace, body):
            journal.append((namespace, body.metadata.name))
            return body

        def read_namespaced_pod(self, name, namespace):
            raise kubernetes.client.exceptions.ApiException(status=404)

        def create_namespaced_pod(self, namespace, body):
            return body

    monkeypatch.setattr(kubernetes.client, "CoreV1Api", _FakeCoreV1Api)
    return journal


def test_image_pull_secret_lands_in_the_workload_namespace(monkeypatch):
    journal = _fake_secret_api(monkeypatch)
    deployer = _deployer()

    secret_name = deployer._apply_image_pull_secret(
        registry="https://index.docker.io/v1/",
        username="u",
        password="p",  # noqa: S106
        namespace="gpustack-acme",
    )

    assert journal == [("gpustack-acme", secret_name)]
    assert deployer._image_pull_secrets == {"gpustack-acme": secret_name}


def test_image_pull_secret_is_copied_once_per_namespace(monkeypatch):
    journal = _fake_secret_api(monkeypatch)
    deployer = _deployer()

    for namespace in ("gpustack-acme", "gpustack-beta", "gpustack-acme"):
        deployer._apply_image_pull_secret(
            registry="https://index.docker.io/v1/",
            username="u",
            password="p",  # noqa: S106
            namespace=namespace,
        )

    # One copy per namespace, and none of them applied twice.
    assert [c[0] for c in journal] == ["gpustack-acme", "gpustack-beta"]


def test_pod_references_the_image_pull_secret_of_its_own_namespace(monkeypatch):
    monkeypatch.setattr(
        "gpustack_runtime.deployer.kuberentes.get_resource_injection_policy",
        lambda *_args: "env",
    )
    monkeypatch.setattr(
        "gpustack_runtime.deployer.kuberentes._resolve_runtime_class_name",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        envs,
        "GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME",
        "u",
    )
    monkeypatch.setattr(
        envs,
        "GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD",
        "p",
    )
    journal = _fake_secret_api(monkeypatch)
    deployer = _deployer()

    workload = KubernetesWorkloadPlan(
        name="test",
        namespace="gpustack-acme",
        containers=[
            Container(
                name="run",
                image="busybox:1.37",
                profile=ContainerProfileEnum.RUN,
            ),
        ],
    )
    workload.validate_and_default()
    pod = deployer._create_pod(workload, {})

    assert [ips.name for ips in pod.spec.image_pull_secrets] == [journal[0][1]]
    assert journal[0][0] == "gpustack-acme"


def _self_pod() -> kubernetes.client.V1Pod:
    return kubernetes.client.V1Pod(
        metadata=kubernetes.client.V1ObjectMeta(
            name="gpustack-agent",
            namespace=_AGENT_NAMESPACE,
        ),
        spec=kubernetes.client.V1PodSpec(
            containers=[
                kubernetes.client.V1Container(
                    name="default",
                    env=[
                        kubernetes.client.V1EnvVar(name="HTTP_PROXY", value="proxy"),
                        kubernetes.client.V1EnvVar(
                            name="HF_TOKEN",
                            value_from=kubernetes.client.V1EnvVarSource(
                                secret_key_ref=kubernetes.client.V1SecretKeySelector(
                                    name="hf",
                                    key="token",
                                ),
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )


def _mirrored_deployer(monkeypatch) -> KubernetesDeployer:
    monkeypatch.setattr(
        KubernetesDeployer,
        "_find_self_pod",
        lambda _self: _self_pod(),
    )

    class _FakeCoreV1Api:
        def __init__(self, client=None):
            pass

    monkeypatch.setattr(kubernetes.client, "CoreV1Api", _FakeCoreV1Api)

    deployer = _deployer()
    deployer._mutate_create_pod = None
    deployer._prepare_mirrored_deployment()
    return deployer


def _mutable_pod(namespace: str) -> kubernetes.client.V1Pod:
    return kubernetes.client.V1Pod(
        metadata=kubernetes.client.V1ObjectMeta(
            name="gpustack-test",
            namespace=namespace,
        ),
        spec=kubernetes.client.V1PodSpec(
            containers=[kubernetes.client.V1Container(name="run", env=[])],
        ),
    )


def test_mirrored_envs_are_mirrored_within_the_worker_namespace(monkeypatch):
    deployer = _mirrored_deployer(monkeypatch)

    pod = deployer._mutate_create_pod(_mutable_pod(_AGENT_NAMESPACE))

    assert [e.name for e in pod.spec.containers[0].env] == ["HTTP_PROXY", "HF_TOKEN"]


def test_mirrored_envs_dropped_outside_the_worker_namespace_are_named(
    monkeypatch,
    caplog,
):
    deployer = _mirrored_deployer(monkeypatch)

    with caplog.at_level(logging.WARNING):
        pod = deployer._mutate_create_pod(_mutable_pod("gpustack-acme"))

    # Kubernetes resolves a Secret reference in the referring Pod's namespace,
    # so the env cannot be mirrored -- but dropping it in silence is what makes
    # a missing token look like a misbehaving workload.
    assert [e.name for e in pod.spec.containers[0].env] == ["HTTP_PROXY"]
    assert "HF_TOKEN" in caplog.text
    assert "gpustack-acme" in caplog.text


def test_mirrored_envs_are_decided_per_pod(monkeypatch):
    # The same deployer serves both namespaces, so the decision cannot be taken
    # once and reused.
    deployer = _mirrored_deployer(monkeypatch)

    acme_pod = deployer._mutate_create_pod(_mutable_pod("gpustack-acme"))
    system_pod = deployer._mutate_create_pod(_mutable_pod(_AGENT_NAMESPACE))

    assert [e.name for e in acme_pod.spec.containers[0].env] == ["HTTP_PROXY"]
    assert [e.name for e in system_pod.spec.containers[0].env] == [
        "HTTP_PROXY",
        "HF_TOKEN",
    ]


def test_mirrored_env_drop_is_reported_once_per_namespace(monkeypatch, caplog):
    deployer = _mirrored_deployer(monkeypatch)

    with caplog.at_level(logging.WARNING):
        deployer._mutate_create_pod(_mutable_pod("gpustack-acme"))
        deployer._mutate_create_pod(_mutable_pod("gpustack-acme"))

    assert caplog.text.count("HF_TOKEN") == 1
