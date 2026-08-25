# The cases below drive the deployers' own deletion path, which is where the
# pre-deletion annotating becomes observable without a live Docker daemon,
# Podman socket or Kubernetes cluster.
# ruff: noqa: SLF001

import logging
from types import MethodType, SimpleNamespace

import kubernetes.client
import kubernetes.client.exceptions
import pytest

import gpustack_runtime.deployer as deployer_pkg
from gpustack_runtime.deployer.__types__ import (
    Deployer,
    OperationError,
)
from gpustack_runtime.deployer.docker import DockerDeployer
from gpustack_runtime.deployer.kuberentes import (
    ANNOTATION_KUEUE_RETRIABLE_IN_GROUP,
    KubernetesDeployer,
)
from gpustack_runtime.deployer.podman import PodmanDeployer

_TEARDOWN = {ANNOTATION_KUEUE_RETRIABLE_IN_GROUP: "false"}
"""
What a caller tearing down a Kueue Pod group stamps: without it the Workload
keeps the group's quota reserved for a replacement that never comes, and the
member stays in Terminating.
"""


def _journalling_deployer(journal: list, annotate_error: Exception | None = None):
    """
    A deployer recording the calls the deletion facade makes on it, in order.
    """

    def _annotate(name, namespace=None, annotations=None):
        journal.append(("annotate", name, namespace, annotations))
        if annotate_error:
            raise annotate_error
        return True

    dep = SimpleNamespace(
        is_supported=lambda: True,
        _annotate=_annotate,
        _delete=lambda *args: journal.append(("delete", *args)),
    )
    # The dispatch under test, bound to the recorder rather than reimplemented.
    dep.annotate = MethodType(Deployer.annotate, dep)
    return dep


def test_annotate_returns_false_without_annotations():
    # Nothing to stamp is not a reason to reach the deployer implementation.
    journal = []
    dep = _journalling_deployer(journal)

    assert Deployer.annotate(dep, name="test", async_mode=False) is False
    assert journal == []


def test_annotate_forwards_the_annotations():
    journal = []
    dep = _journalling_deployer(journal)

    assert (
        Deployer.annotate(
            dep,
            name="test",
            annotations=_TEARDOWN,
            async_mode=False,
        )
        is True
    )
    assert journal == [("annotate", "test", None, _TEARDOWN)]


def test_delete_annotates_before_deleting():
    # The whole point of the parameter: the annotation informs the controller
    # reacting to the deletion, so it has to be there before the deletion is.
    journal = []
    dep = _journalling_deployer(journal)

    Deployer.delete(
        dep,
        name="test",
        namespace="default",
        annotations=_TEARDOWN,
        async_mode=False,
    )

    assert journal == [
        ("annotate", "test", "default", _TEARDOWN),
        ("delete", "test", "default", None),
    ]


def test_delete_without_annotations_does_not_annotate():
    # The deletion path every existing caller takes is left untouched.
    journal = []
    dep = _journalling_deployer(journal)

    Deployer.delete(dep, name="test", async_mode=False)

    assert journal == [("delete", "test", None, None)]


def test_delete_survives_a_failing_annotation(caplog):
    # A workload that keeps running holds the devices it was deployed with,
    # which is worse than a controller reading a stale annotation, so the
    # deletion carries on -- loudly.
    journal = []
    dep = _journalling_deployer(journal, annotate_error=OperationError("boom"))

    with caplog.at_level(logging.WARNING):
        Deployer.delete(dep, name="test", annotations=_TEARDOWN, async_mode=False)

    assert journal == [
        ("annotate", "test", None, _TEARDOWN),
        ("delete", "test", None, None),
    ]
    assert "Failed to annotate workload test" in caplog.text


@pytest.mark.parametrize("deployer_cls", [DockerDeployer, PodmanDeployer])
def test_annotating_is_a_logged_no_op_off_kubernetes(deployer_cls, caplog):
    # There is no Kueue on a container runtime, so there is nothing to tell:
    # the request is refused rather than reported as honored.
    deployer = object.__new__(deployer_cls)
    Deployer.__init__(deployer, "test")

    with caplog.at_level(logging.WARNING):
        annotated = deployer._annotate("test", None, _TEARDOWN)

    assert annotated is False
    assert "cannot annotate workload test" in caplog.text


@pytest.mark.parametrize("deployer_cls", [DockerDeployer, PodmanDeployer])
def test_delete_off_kubernetes_still_deletes(deployer_cls, caplog):
    deployer = object.__new__(deployer_cls)
    Deployer.__init__(deployer, "test")
    calls = []
    deployer._delete = lambda *args: calls.append(args)

    with caplog.at_level(logging.WARNING):
        Deployer.delete(
            deployer,
            name="test",
            annotations=_TEARDOWN,
            async_mode=False,
        )

    assert calls == [("test", None, None)]
    assert "cannot annotate workload test" in caplog.text


class _FakeCoreV1AnnotateApi:
    """
    Stand-in for the Kubernetes core API, recording the calls it receives.
    """

    def __init__(self, journal: list, patch_error: Exception | None = None):
        self.journal = journal
        self.patch_error = patch_error

    def __call__(self, client=None):
        return self

    def patch_namespaced_pod(self, name, namespace, body):
        self.journal.append(("patch", name, namespace, body))
        if self.patch_error:
            raise self.patch_error

    def delete_collection_namespaced_pod(self, **_kwargs):
        self.journal.append(("delete_pods",))

    def delete_collection_namespaced_service(self, **_kwargs):
        self.journal.append(("delete_services",))

    def delete_collection_namespaced_config_map(self, **_kwargs):
        self.journal.append(("delete_configmaps",))


def _pods(*namespaced_names) -> list:
    return [
        SimpleNamespace(
            metadata=SimpleNamespace(namespace=namespace, name=name),
        )
        for namespace, name in namespaced_names
    ]


def _kubernetes_deployer(
    monkeypatch,
    journal: list,
    pods: list,
    patch_error: Exception | None = None,
) -> KubernetesDeployer:
    monkeypatch.setattr(
        kubernetes.client,
        "CoreV1Api",
        _FakeCoreV1AnnotateApi(journal, patch_error),
    )

    deployer = object.__new__(KubernetesDeployer)
    Deployer.__init__(deployer, "kubernetes")
    deployer._client = None
    deployer.is_supported = lambda: True
    deployer._list_workload_pods = lambda **_kwargs: pods
    # The status lookup the deletion starts with is not what is under test.
    deployer.get = lambda **_kwargs: SimpleNamespace(
        name="test",
        namespace="gpustack-default",
    )
    return deployer


def test_kubernetes_annotate_patches_every_pod_of_the_workload(monkeypatch):
    journal = []
    deployer = _kubernetes_deployer(
        monkeypatch,
        journal,
        _pods(("gpustack-default", "test-0"), ("gpustack-default", "test-1")),
    )

    assert deployer._annotate("test", None, _TEARDOWN) is True
    assert journal == [
        (
            "patch",
            "test-0",
            "gpustack-default",
            {"metadata": {"annotations": _TEARDOWN}},
        ),
        (
            "patch",
            "test-1",
            "gpustack-default",
            {"metadata": {"annotations": _TEARDOWN}},
        ),
    ]


def test_kubernetes_annotate_is_a_no_op_for_a_missing_workload(monkeypatch):
    # Deleting is idempotent, so annotating on the way to it must be too.
    journal = []
    deployer = _kubernetes_deployer(monkeypatch, journal, [])

    assert deployer._annotate("test", None, _TEARDOWN) is False
    assert journal == []


def test_kubernetes_annotate_skips_a_pod_deleted_underneath_it(monkeypatch):
    # A Pod gone between the lookup and the patch needs no annotation either.
    journal = []
    deployer = _kubernetes_deployer(
        monkeypatch,
        journal,
        _pods(("gpustack-default", "test-0")),
        patch_error=kubernetes.client.exceptions.ApiException(status=404),
    )

    assert deployer._annotate("test", None, _TEARDOWN) is False


def test_kubernetes_annotate_reports_a_failure_as_an_operation_error(monkeypatch):
    journal = []
    deployer = _kubernetes_deployer(
        monkeypatch,
        journal,
        _pods(("gpustack-default", "test-0")),
        patch_error=kubernetes.client.exceptions.ApiException(status=403),
    )

    with pytest.raises(OperationError):
        deployer._annotate("test", None, _TEARDOWN)


def test_kubernetes_annotate_reports_a_transport_failure_as_an_operation_error(
    monkeypatch,
):
    # A transport failure carries no HTTP status, so it must not escape the
    # annotating as something other than an OperationError.
    journal = []
    deployer = _kubernetes_deployer(
        monkeypatch,
        journal,
        _pods(("gpustack-default", "test-0")),
        patch_error=OSError("connection reset"),
    )

    with pytest.raises(OperationError):
        deployer._annotate("test", None, _TEARDOWN)


def test_kubernetes_delete_patches_the_pods_before_removing_them(monkeypatch):
    journal = []
    deployer = _kubernetes_deployer(
        monkeypatch,
        journal,
        _pods(("gpustack-default", "test-0")),
    )

    Deployer.delete(
        deployer,
        name="test",
        namespace="gpustack-default",
        annotations=_TEARDOWN,
        async_mode=False,
    )

    assert journal[0][0] == "patch"
    assert ("delete_pods",) in journal


def test_kubernetes_delete_removes_the_pods_despite_a_failing_patch(monkeypatch):
    journal = []
    deployer = _kubernetes_deployer(
        monkeypatch,
        journal,
        _pods(("gpustack-default", "test-0")),
        patch_error=kubernetes.client.exceptions.ApiException(status=403),
    )

    Deployer.delete(
        deployer,
        name="test",
        namespace="gpustack-default",
        annotations=_TEARDOWN,
        async_mode=False,
    )

    assert ("delete_pods",) in journal


def test_delete_workload_forwards_the_annotations(monkeypatch):
    calls = []
    dep = SimpleNamespace(
        is_supported=lambda: True,
        delete=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(deployer_pkg, "_DEPLOYERS", [dep])

    deployer_pkg.delete_workload("test", annotations=_TEARDOWN)

    assert calls == [
        {
            "name": "test",
            "namespace": None,
            "grace_period_seconds": None,
            "annotations": _TEARDOWN,
        },
    ]


def test_delete_workload_keeps_its_existing_callers(monkeypatch):
    # Every existing call site passes the name only, positionally.
    calls = []
    dep = SimpleNamespace(
        is_supported=lambda: True,
        delete=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(deployer_pkg, "_DEPLOYERS", [dep])

    deployer_pkg.delete_workload("test")

    assert calls[0]["annotations"] is None
