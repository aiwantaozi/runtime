# The cases below drive the Pod reconciliation path: comparing an actual Pod
# against the desired one, which decides whether the Pod is recreated, and the
# declarations that comparison reads.
# ruff: noqa: SLF001

import kubernetes.client
import kubernetes.watch

from gpustack_runtime.deployer.__types__ import (
    Container,
    ContainerProfileEnum,
    Deployer,
)
from gpustack_runtime.deployer.kuberentes import (
    _ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT,
    _LABEL_COMPONENT,
    _LABEL_KUEUE_POD_GROUP_NAME,
    _LABEL_KUEUE_QUEUE_NAME,
    _LABEL_WORKLOAD,
    _WATCH_TIMEOUT_SECONDS,
    KubernetesDeployer,
    KubernetesWorkloadPlan,
    equal_pods,
    watch,
)


def _pod(
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    node_selector: dict[str, str] | None = None,
) -> kubernetes.client.V1Pod:
    return kubernetes.client.V1Pod(
        metadata=kubernetes.client.V1ObjectMeta(
            name="gpustack-test",
            namespace="gpustack-default",
            labels={_LABEL_WORKLOAD: "test", **(labels or {})},
            annotations={
                f"{_LABEL_COMPONENT}-run-0-name": "run",
                **(annotations or {}),
            },
        ),
        spec=kubernetes.client.V1PodSpec(
            containers=[
                kubernetes.client.V1Container(
                    name="run",
                    image="busybox:1.37",
                ),
            ],
            node_selector=node_selector,
            restart_policy="Always",
            termination_grace_period_seconds=30,
        ),
    )


def test_equal_pods_of_the_same_declaration():
    assert equal_pods(_pod(), _pod())


def test_equal_pods_ignores_the_metadata_it_does_not_declare():
    # The API server, the webhooks and the controllers taking the Pod over all
    # stamp their own metadata on it, and none of it says anything about the
    # workload: comparing it would recreate the Pod on every reconcile.
    actual = _pod(
        labels={"app": "something-else"},
        annotations={
            "kubectl.kubernetes.io/last-applied-configuration": "{}",
            "cni.projectcalico.org/podIP": "10.0.0.1/32",
        },
    )

    assert equal_pods(actual, _pod())


def test_equal_pods_detects_added_gang_markers():
    # Gang admission lives in the metadata alone, so a Pod matching on the spec
    # would keep running ungrouped.
    actual = _pod()
    desired = _pod(
        labels={_LABEL_KUEUE_POD_GROUP_NAME: "test"},
        annotations={_ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT: "2"},
    )

    assert not equal_pods(actual, desired)


def test_equal_pods_detects_a_changed_gang_size():
    actual = _pod(
        labels={_LABEL_KUEUE_POD_GROUP_NAME: "test"},
        annotations={_ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT: "2"},
    )
    desired = _pod(
        labels={_LABEL_KUEUE_POD_GROUP_NAME: "test"},
        annotations={_ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT: "3"},
    )

    assert not equal_pods(actual, desired)


def test_equal_pods_detects_an_added_queue():
    actual = _pod()
    desired = _pod(labels={_LABEL_KUEUE_QUEUE_NAME: "gpustack-fnv64-0"})

    assert not equal_pods(actual, desired)


def test_equal_pods_ignores_a_foreign_managed_key_only_the_actual_carries():
    # An admission webhook may default or add the keys of the controller
    # admitting the workload, e.g. Kueue defaulting the queue name, and
    # comparing those both ways would recreate the Pod forever.
    actual = _pod(
        labels={
            _LABEL_KUEUE_QUEUE_NAME: "default",
            "kueue.x-k8s.io/managed": "true",
        },
    )

    assert equal_pods(actual, _pod())


def test_equal_pods_detects_a_dropped_own_annotation():
    # Nothing but this deployer writes its own keys, so a difference in either
    # direction is one it made, e.g. renaming or dropping a container.
    actual = _pod(annotations={f"{_LABEL_COMPONENT}-init-0-name": "init"})

    assert not equal_pods(actual, _pod())


def test_equal_pods_detects_a_renamed_container_annotation():
    actual = _pod()
    desired = _pod()
    desired.metadata.annotations[f"{_LABEL_COMPONENT}-run-0-name"] = "serve"

    assert not equal_pods(actual, desired)


def test_equal_pods_requires_the_declared_node_selector():
    # A Kueue-gated Pod is pinned by node selector rather than by node name,
    # so an unpinned Pod is not the declared one.
    actual = _pod()
    desired = _pod(node_selector={"kubernetes.io/hostname": "node-0"})

    assert not equal_pods(actual, desired)


def test_equal_pods_detects_a_repinned_node_selector():
    actual = _pod(node_selector={"kubernetes.io/hostname": "node-0"})
    desired = _pod(node_selector={"kubernetes.io/hostname": "node-1"})

    assert not equal_pods(actual, desired)


def test_equal_pods_ignores_the_node_selectors_admission_adds():
    # Kueue copies the node labels of the ResourceFlavor it admitted the
    # workload on into the Pod.
    actual = _pod(
        node_selector={
            "kubernetes.io/hostname": "node-0",
            "nvidia.com/gpu.product": "H100",
        },
    )
    desired = _pod(node_selector={"kubernetes.io/hostname": "node-0"})

    assert equal_pods(actual, desired)


def test_equal_pods_survives_a_pod_carrying_no_metadata_at_all():
    actual = _pod()
    actual.metadata.labels = None
    actual.metadata.annotations = None
    desired = _pod()
    desired.metadata.labels = None
    desired.metadata.annotations = None

    assert equal_pods(actual, desired)


def test_pod_carries_the_declared_annotations(monkeypatch):
    # The gang markers a workload is admitted by live in the annotations, so
    # the plan must be able to declare them.
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

    annotations = {_ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT: "2"}
    workload = KubernetesWorkloadPlan(
        name="test",
        namespace="gpustack-default",
        annotations=annotations,
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

    assert pod.metadata.annotations[_ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT] == "2"
    # The conversion stamps its own annotations onto the Pod, e.g. the
    # container names, and must not write them back into the plan.
    assert annotations == {_ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT: "2"}


def _fake_watch(monkeypatch) -> dict:
    """
    Patch the watch so that streaming records the options it receives.
    """
    captured = {}

    class _FakeWatch:
        def stream(self, func, *args, **kwargs):
            captured.update(kwargs)
            return iter(())

        def stop(self):
            pass

    monkeypatch.setattr(kubernetes.watch, "Watch", _FakeWatch)
    return captured


def test_watch_is_bounded(monkeypatch):
    # Recreating a Pod waits for its deletion on a watch: an unbounded one
    # turns a dropped connection, or an event that never comes, into a
    # deployment waiting forever.
    captured = _fake_watch(monkeypatch)

    with watch(lambda **_kwargs: None, namespace="gpustack-default") as es:
        assert list(es) == []

    assert captured["timeout_seconds"] == _WATCH_TIMEOUT_SECONDS


def test_watch_keeps_a_declared_timeout(monkeypatch):
    captured = _fake_watch(monkeypatch)

    with watch(lambda **_kwargs: None, timeout_seconds=5) as es:
        assert list(es) == []

    assert captured["timeout_seconds"] == 5
