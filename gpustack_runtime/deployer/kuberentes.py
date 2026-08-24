from __future__ import annotations as __future_annotations__

import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import kubernetes
import kubernetes.stream.ws_client
import urllib3.connection
from cachetools.func import ttl_cache
from dataclasses_json import dataclass_json
from gpustack_runner import parse_image

from .. import envs
from ..logging import debug_log_exception
from . import ContainerCheck, ContainerMountModeEnum, OperationError
from .__types__ import (
    DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS,
    Container,
    ContainerPort,
    ContainerProfileEnum,
    ContainerResources,
    EndoscopicDeployer,
    UnsupportedError,
    WorkloadExecStream,
    WorkloadName,
    WorkloadNamespace,
    WorkloadOperationToken,
    WorkloadPlan,
    WorkloadStatus,
    WorkloadStatusExit,
    WorkloadStatusOperation,
    WorkloadStatusStateEnum,
)
from .__utils__ import (
    base64_encode,
    fnv1a_32_hex,
    fnv1a_64_hex,
    safe_yaml,
    sensitive_env_var,
    validate_rfc1123_domain_name,
)
from .k8s.devicemanager import get_resource_injection_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)
clogger = logger.getChild("conversion")

_LABEL_WORKLOAD = f"{envs.GPUSTACK_RUNTIME_DEPLOY_LABEL_PREFIX}/workload"
_LABEL_COMPONENT = f"{envs.GPUSTACK_RUNTIME_DEPLOY_LABEL_PREFIX}/component"

_LABEL_KUEUE_QUEUE_NAME = "kueue.x-k8s.io/queue-name"
"""
Label naming the Kueue LocalQueue admitting the Pod.
"""
_LABEL_KUEUE_POD_GROUP_NAME = "kueue.x-k8s.io/pod-group-name"
"""
Label naming the Kueue Pod group the Pod belongs to,
i.e. the gang admitted as a whole.
"""
_ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT = "kueue.x-k8s.io/pod-group-total-count"
"""
Annotation carrying how many Pods the Kueue Pod group counts,
which Kueue waits for before admitting the gang.
"""
_ANNOTATION_PREFERRED_DEVICE_INDEX = "device.gpustack.ai/accelerator.preferred-index"
"""
Annotation selecting the devices the GPUStack device plugin allocates.
"""

_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

_WORKLOAD_NAMESPACE_PREFIX = "gpustack-"
"""
Prefix of the namespaces holding workloads, one per organization, e.g.
"gpustack-default". A worker hosts the workloads of several organizations, so
the namespace belongs to the workload rather than to the deployer, and an
operation carrying a name only searches these namespaces instead of assuming
the namespace the deployer itself runs in -- which is also the namespace Kueue
structurally cannot admit workloads in, as it excludes its own.
"""


def _default_workload_namespace() -> WorkloadNamespace:
    """
    Return the namespace a workload lands in when it declares none.

    This is the single source of the namespace fallback: every operation
    resolving a workload namespace ends up here, so a deployment declaring no
    namespace anywhere keeps deploying, reading and deleting where it always
    has.

    Returns:
        The configured default namespace.

    """
    return envs.GPUSTACK_RUNTIME_KUBERNETES_NAMESPACE


def _is_workload_namespace(namespace: WorkloadNamespace | None) -> bool:
    """
    Report whether the given namespace may hold workloads, which the
    per-organization namespaces do, and so does the configured default: a
    single-namespace deployment holds its workloads there.

    Args:
        namespace:
            The namespace to judge.

    Returns:
        True if the namespace may hold workloads, False otherwise.

    """
    if not namespace:
        return False
    return (
        namespace.startswith(_WORKLOAD_NAMESPACE_PREFIX)
        or namespace == _default_workload_namespace()
    )


_IMAGE_PULL_BLOCKED_REASONS = frozenset(
    {
        "ErrImageNeverPull",
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
        "RegistryUnavailable",
    },
)
"""
Container waiting reasons meaning the image can never be pulled as requested,
so the workload has failed instead of merely waiting to start.
"""

_IMAGE_PULL_FAILED_EVENT_REASON = "Failed"
"""
The reason a kubelet stamps on the Event carrying the registry error, e.g.
"Failed to pull image ...: unauthorized". Any other warning the Pod collected,
a stale FailedScheduling among them, explains something else.
"""


class KubernetesWorkloadServiceTypeEnum(str, Enum):
    """
    Types for Kubernetes Service.
    """

    CLUSTER_IP = "ClusterIP"
    """
    ClusterIP: Exposes the service on a cluster-internal IP.
    """
    NODE_PORT = "NodePort"
    """
    NodePort: Exposes the service on each Node's IP at a static port.
    """
    LOAD_BALANCER = "LoadBalancer"
    """
    LoadBalancer: Exposes the service externally using a cloud provider's load balancer.
    """

    def __str__(self):
        return self.value


@dataclass_json
@dataclass
class KubernetesWorkloadPlan(WorkloadPlan):
    """
    Workload plan implementation for Kubernetes Deployment.

    Attributes:
        domain_suffix (str):
            Domain suffix for the cluster. Default is "cluster.local".
        service_type (KubernetesWorkloadServiceTypeEnum):
            Service type for the workload. Default is CLUSTER_IP.
        namespace (str | None):
            Namespace of the workload.
        name (str):
            Name of the workload,
            it should be unique in the deployer.
        labels (dict[str, str] | None):
            Labels to attach to the workload.
        annotations (dict[str, str] | None):
            Annotations to attach to the workload,
            which land on the Pod.
        host_network (bool):
            Indicates if the containers of the workload use the host network.
        host_ipc (bool):
            Indicates if the containers of the workload use the host IPC.
        pid_shared (bool):
            Indicates if the containers of the workload share the PID namespace.
        shm_size (int | str | None):
            Configure shared memory size for the workload.
        run_as_user (int | None):
            The user ID to run the workload as.
        run_as_group (int | None):
            The group ID to run the workload as.
        fs_group (int | None):
            The group ID to own the filesystem of the workload.
        sysctls (dict[str, str] | None):
            Sysctls to set for the workload.
        termination_grace_period_seconds (int):
            Duration in seconds the containers of the workload need to terminate gracefully.
        containers (list[tuple[int, Container]] | None):
            List of containers in the workload.
            It must contain at least one "RUN" profile container.

    """

    domain_suffix: str = envs.GPUSTACK_RUNTIME_KUBERNETES_DOMAIN_SUFFIX
    """
    Domain suffix for the cluster.
    """
    service_type: KubernetesWorkloadServiceTypeEnum = field(
        default_factory=lambda: envs.GPUSTACK_RUNTIME_KUBERNETES_SERVICE_TYPE,
    )
    """
    Service type for the workload.
    """

    def validate_and_default(self):
        """
        Validate and set defaults for the workload plan.

        Raises:
            ValueError:
                If the workload plan is invalid.

        """
        if self.labels is None:
            self.labels = {}
        if self.containers is None:
            self.containers = []

        self.labels[_LABEL_WORKLOAD] = self.name

        # Default and validate in the base class.
        super().validate_and_default()

        # Validate namespace
        if not self.namespace:
            self.namespace = _default_workload_namespace()
        try:
            validate_rfc1123_domain_name(self.namespace)
        except ValueError as e:
            msg = f"Invalid namespace '{self.namespace}'"
            raise ValueError(msg) from e


@dataclass_json
@dataclass
class KubernetesWorkloadStatus(WorkloadStatus):
    """
    Workload status implementation for Kubernetes Deployment.
    """

    _k_pod: kubernetes.client.V1Pod | None = field(
        default=None,
        repr=False,
        metadata={
            "dataclasses_json": {
                "exclude": lambda _: True,
                "encoder": lambda _: None,
                "decoder": lambda _: None,
            },
        },
    )
    """
    Pod object from Kubernetes API,
    internal use only.
    """

    @staticmethod
    def parse_state(
        k_pod: kubernetes.client.V1Pod,
    ) -> WorkloadStatusStateEnum:
        """
        Parse the state of the workload from the Kubernetes Pod object.

        Args:
            k_pod:
                Pod object from Kubernetes API.

        Returns:
            The state of the workload.

        """
        match k_pod.status.phase:
            case "Pending":
                # A Pod blocked on an image it can never pull is not waiting for
                # anything: report it as failed so the caller stops watching a
                # workload that will never become ready.
                if KubernetesWorkloadStatus.parse_image_pull_block(k_pod):
                    return WorkloadStatusStateEnum.FAILED
                return WorkloadStatusStateEnum.PENDING
            case "Succeeded":
                return WorkloadStatusStateEnum.INACTIVE
            case "Failed":
                return WorkloadStatusStateEnum.FAILED
            case "Unknown":
                return WorkloadStatusStateEnum.UNKNOWN
            case "Running":
                if not k_pod.status.container_statuses:
                    return WorkloadStatusStateEnum.INITIALIZING
                for cs in k_pod.status.init_container_statuses or []:
                    if not cs.ready:
                        return WorkloadStatusStateEnum.INITIALIZING
                for cs in k_pod.status.container_statuses:
                    if not cs.ready:
                        return WorkloadStatusStateEnum.INITIALIZING

        return WorkloadStatusStateEnum.RUNNING

    @staticmethod
    def parse_image_pull_block(
        k_pod: kubernetes.client.V1Pod,
    ) -> kubernetes.client.V1ContainerStateWaiting | None:
        """
        Find the container state blocking the workload on an image pull.

        Args:
            k_pod:
                Pod object from Kubernetes API.

        Returns:
            The waiting state of the first container blocked on its image pull,
            None if no container is.

        """
        for cs in [
            *(k_pod.status.init_container_statuses or []),
            *(k_pod.status.container_statuses or []),
        ]:
            waiting = cs.state.waiting if cs.state else None
            if waiting and waiting.reason in _IMAGE_PULL_BLOCKED_REASONS:
                return waiting
        return None

    @staticmethod
    def _parse_exit(
        cs: kubernetes.client.V1ContainerStatus | None,
        name: str,
    ) -> WorkloadStatusExit | None:
        """
        Build the exit entry for a container that has terminated or is blocked
        from starting.

        Args:
            cs:
                Status of the container, None if the Pod reports none yet.
            name:
                Human-readable name of the container.

        Returns:
            A WorkloadStatusExit if the container has terminated or is blocked,
            None if it is running or has no state at all, and so contributes
            nothing.

        """
        if not cs or not cs.state:
            return None

        # Prefer the current termination, and fall back to the previous one, so
        # a container that has already restarted still reports why it died.
        terminated = cs.state.terminated or (
            cs.last_state.terminated if cs.last_state else None
        )
        if terminated:
            return WorkloadStatusExit(
                name=name,
                token=cs.name,
                exit_code=terminated.exit_code,
                reason=terminated.reason or "",
                message=terminated.message or "",
                started_at=(
                    terminated.started_at.strftime(_TIMESTAMP_FORMAT)
                    if terminated.started_at
                    else ""
                ),
                finished_at=(
                    terminated.finished_at.strftime(_TIMESTAMP_FORMAT)
                    if terminated.finished_at
                    else ""
                ),
                restart_count=cs.restart_count or 0,
            )

        if cs.state.waiting:
            # A container blocked from starting has never terminated,
            # so it carries a reason but no exit code.
            return WorkloadStatusExit(
                name=name,
                token=cs.name,
                reason=cs.state.waiting.reason or "",
                message=cs.state.waiting.message or "",
                restart_count=cs.restart_count or 0,
            )

        return None

    @staticmethod
    def _parse_pod_event_message(
        k_pod: kubernetes.client.V1Pod,
        core_api: kubernetes.client.CoreV1Api,
    ) -> str:
        """
        Read the Events of the given Pod and return the most relevant message.

        Args:
            k_pod:
                Pod object from Kubernetes API.
            core_api:
                Core API client to read the Events with.

        Returns:
            The message of the Pod's latest warning Event,
            empty if there is none or if the Events cannot be read.

        """
        field_selector = f"involvedObject.name={k_pod.metadata.name}"
        if k_pod.metadata.uid:
            # A recreated Pod takes the same name, so selecting by name alone
            # also selects its predecessor's Events.
            field_selector += f",involvedObject.uid={k_pod.metadata.uid}"

        try:
            k_events = core_api.list_namespaced_event(
                namespace=k_pod.metadata.namespace,
                field_selector=field_selector,
            )
        except kubernetes.client.exceptions.ApiException:
            # Any API failure degrades to no message rather than propagating:
            # this read only enriches a diagnosis the state and the reason
            # already carry. The expected one is a 403 on a cluster whose
            # manifest predates the Events rule, but a 500 or a timeout is no
            # reason to fail a status poll either.
            debug_log_exception(
                logger,
                "Failed to list events of pod %s/%s",
                k_pod.metadata.namespace,
                k_pod.metadata.name,
            )
            return ""

        # The warning Events carry the registry error, e.g. the "Failed" Event
        # behind a bare "ImagePullBackOff" reason.
        k_warnings = [
            k_event
            for k_event in k_events.items or []
            if k_event.type == "Warning" and k_event.message
        ]
        if not k_warnings:
            return ""

        # The API guarantees no ordering and an Event name carries a random
        # suffix, so list order is not time order: sort it. An Event carrying no
        # timestamp at all sorts first, i.e. loses to any that does.
        k_warnings.sort(
            key=lambda e: (e.last_timestamp or e.event_time or e.first_timestamp)
            or datetime.min.replace(tzinfo=timezone.utc),
        )

        # Among the warnings, the kubelet's image-pull failure is the one being
        # diagnosed, so it wins over a later warning about something else.
        return next(
            (
                k_event.message
                for k_event in reversed(k_warnings)
                if k_event.reason == _IMAGE_PULL_FAILED_EVENT_REASON
            ),
            k_warnings[-1].message,
        )

    def __init__(
        self,
        name: WorkloadName,
        k_pod: kubernetes.client.V1Pod,
        core_api: kubernetes.client.CoreV1Api | None = None,
        **kwargs,
    ):
        created_at = k_pod.metadata.creation_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        labels = {
            k: v
            for k, v in (k_pod.metadata.labels or {}).items()
            if not k.startswith("runtime.gpustack.ai/")
        }
        annotations = {
            k: v
            for k, v in (k_pod.metadata.annotations or {}).items()
            if not k.startswith("runtime.gpustack.ai/")
        }

        super().__init__(
            name=name,
            created_at=created_at,
            namespace=k_pod.metadata.namespace,
            labels=labels,
            annotations=annotations,
            **kwargs,
        )

        self._k_pod = k_pod

        k_init_css = {cs.name: cs for cs in k_pod.status.init_container_statuses or []}
        k_css = {cs.name: cs for cs in k_pod.status.container_statuses or []}

        k_pod_annos = k_pod.metadata.annotations or {}
        for ci, c in enumerate(k_pod.spec.init_containers or []):
            cn = k_pod_annos.get(f"{_LABEL_COMPONENT}-init-{ci}-name", c.name)
            op = WorkloadStatusOperation(
                name=cn,
                token=c.name,
            )
            if c.restart_policy != "Never":
                self.executable.append(op)
            self.loggable.append(op)

            exit_ = self._parse_exit(k_init_css.get(c.name), cn)
            if exit_:
                self.exits.append(exit_)

        for ci, c in enumerate(k_pod.spec.containers):
            cn = k_pod_annos.get(f"{_LABEL_COMPONENT}-run-{ci}-name", c.name)
            op = WorkloadStatusOperation(
                name=cn,
                token=c.name,
            )
            self.executable.append(op)
            self.loggable.append(op)

            exit_ = self._parse_exit(k_css.get(c.name), cn)
            if exit_:
                self.exits.append(exit_)

        self.state = self.parse_state(k_pod)

        # Only a Pod failed on an image pull reads Events, which is both the
        # verdict's own condition and the traffic guard: any other Pod, running
        # or pending, costs no extra API call.
        image_pull_block = (
            self.parse_image_pull_block(k_pod)
            if self.state == WorkloadStatusStateEnum.FAILED
            and k_pod.status.phase == "Pending"
            else None
        )
        if image_pull_block:
            # The waiting reason alone ("ImagePullBackOff") omits the registry
            # error, which is the part explaining why the pull fails, so append
            # the Pod's Event message to it. This diagnosis is more specific
            # than the Pod's own status message, hence it wins.
            blocked_message = ": ".join(
                p for p in [image_pull_block.reason, image_pull_block.message] if p
            )
            event_message = (
                self._parse_pod_event_message(k_pod, core_api) if core_api else ""
            )
            self.state_message = (
                f"{blocked_message}; {event_message}"
                if event_message
                else blocked_message
            )
        elif k_pod.status.message:
            # Surface the Pod's status message, e.g. a device-plugin
            # admission rejection ("UnexpectedAdmissionError: Allocate
            # failed ...") on Failed Pods.
            self.state_message = k_pod.status.message


_NAME = "kubernetes"
"""
Name of the Kubernetes deployer.
"""


def _apply_instance_type_admission(
    pod: kubernetes.client.V1Pod,
    instance_type: str | None,
) -> None:
    """
    Stamp the Kueue queue-name label of the InstanceType's entrance
    LocalQueue onto the Pod, so the workload is admitted through the
    operator's queue for that InstanceType. The LocalQueue name is the
    FNV-1a 64 hash of the InstanceType name, mirroring the operator's
    FormatLocalQueueName ("gpustack-fnv64-<fnv64a-hex>").

    Skipped when no InstanceType is set, or when Kueue admission is
    disabled via GPUSTACK_RUNTIME_KUBERNETES_KDP_NO_KUEUE_ADMISSION —
    then the Pod is scheduled by the plain Kubernetes scheduler.
    """
    if not instance_type:
        return
    if envs.GPUSTACK_RUNTIME_KUBERNETES_KDP_NO_KUEUE_ADMISSION:
        return
    pod.metadata.labels[_LABEL_KUEUE_QUEUE_NAME] = (
        f"gpustack-fnv64-{fnv1a_64_hex(instance_type)}"
    )


def _pin_pod_for_kueue(
    pod: kubernetes.client.V1Pod,
    node_name: str | None,
) -> None:
    """
    Adjust node pinning for Kueue-gated Pods in place.

    Kueue-gated Pods (queue label present) cannot set spec.nodeName:
    the API forbids nodeName until all schedulingGates have been
    cleared. Pin via nodeSelector instead; kube-scheduler places
    the Pod on the same node once Kueue admits it.
    """
    if not node_name:
        return
    if _LABEL_KUEUE_QUEUE_NAME not in (pod.metadata.labels or {}):
        return
    pod.spec.node_name = None
    pod.spec.node_selector = {
        **(pod.spec.node_selector or {}),
        "kubernetes.io/hostname": node_name,
    }


def _is_device_plugin_resource(resource_key: str) -> bool:
    """
    Report whether a resource key belongs to a device plugin resource family,
    which is either a CDI kind ("nvidia.com/gpu") or one of its suffixed
    variants ("nvidia.com/gpu.shared", "nvidia.com/gpu.sliced.units",
    "nvidia.com/gpu.partitioned.mig-1g.20gb").
    """
    return any(
        resource_key == cdi or resource_key.startswith(f"{cdi}.")
        for cdi in envs.GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI.values()
    )


def _is_overcommittable_resource(resource_key: str) -> bool:
    """
    Report whether a resource key may request and limit different quantities,
    which only the native compute resources may: Kubernetes requires an
    extended resource -- every device plugin resource among them -- to request
    and limit the same quantity, and rejects the Pod declaring otherwise.
    """
    return resource_key in (
        "cpu",
        "memory",
        "ephemeral-storage",
    ) or resource_key.startswith("hugepages-")


def _container_resource_requirements(
    container_name: str,
    requests: dict[str, str],
    limits: ContainerResources | None,
) -> kubernetes.client.V1ResourceRequirements:
    """
    Build the resource requirements of a Container from the resources it
    requests and the resources it is limited to.

    A container declaring no limits of its own is limited to what it requests,
    which makes it Guaranteed, as every container has been so far. Declaring
    them apart makes it Burstable, which is the point of declaring them at all.

    Args:
        container_name:
            The name of the container, for reporting.
        requests:
            The resources the container requests, as converted for the Pod.
        limits:
            The resources the container is limited to, as declared.

    Returns:
        The resource requirements of the container.

    """
    if not requests:
        return kubernetes.client.V1ResourceRequirements(
            limits=None,
            requests=None,
        )

    resolved_limits = dict(requests)
    for l_k, l_v in (limits or {}).items():
        if l_k not in resolved_limits:
            # A mapped device request never reaches the Pod as a resource, e.g.
            # under the env injection policy it becomes a visible-devices env,
            # so a limit on it has nothing to limit.
            continue
        l_v = str(l_v)  # noqa: PLW2901
        if l_v == resolved_limits[l_k]:
            continue
        if not _is_overcommittable_resource(l_k):
            clogger.warning(
                "Container '%s' limits resource '%s' to %s but requests %s, "
                "which Kubernetes rejects for an extended resource: "
                "limiting it to what it requests",
                container_name,
                l_k,
                l_v,
                resolved_limits[l_k],
            )
            continue
        resolved_limits[l_k] = l_v

    return kubernetes.client.V1ResourceRequirements(
        limits=resolved_limits,
        requests=dict(requests),
    )


def _match_runtime_class(resource_key: str) -> str:
    """
    Return the RuntimeClass name mapped to the device plugin resource family
    of the given resource key, which matches either a mapped base key
    ("huawei.com/npu") or one of its suffixed variants
    ("huawei.com/npu.sliced", "huawei.com/npu.sliced.units").
    Return an empty string if the key belongs to no mapped family.
    """
    for base_key, runtime_class_name in (
        envs.GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_CLASS or {}
    ).items():
        if resource_key == base_key or resource_key.startswith(f"{base_key}."):
            return runtime_class_name
    return ""


def _resolve_runtime_class_name(
    pod: kubernetes.client.V1Pod,
    client: kubernetes.client.ApiClient,
):
    """
    Resolve the RuntimeClass for the Pod from the device plugin resources
    requested by its containers, and set `pod.spec.runtime_class_name`
    when the mapped RuntimeClass object exists in the cluster.

    A Pod that already names a runtime class is left untouched, and the
    mirrored deployment fill stays the fallback for Pods without any
    mapped device request, so the precedence is:
    explicit > discovered > mirrored.
    """
    if pod.spec.runtime_class_name:
        return

    # Find the RuntimeClass mapped to the requested device resources,
    # keeping the first match on multi-vendor conflicts.
    runtime_class_name = ""
    for container in (pod.spec.containers or []) + (pod.spec.init_containers or []):
        resources = container.resources
        if not resources or not resources.requests:
            continue
        for r_k in resources.requests:
            matched = _match_runtime_class(r_k)
            if not matched:
                continue
            if runtime_class_name and runtime_class_name != matched:
                logger.warning(
                    "Container '%s' requests resource '%s' mapped to RuntimeClass '%s', "
                    "but keeping the first matched RuntimeClass '%s'",
                    container.name,
                    r_k,
                    matched,
                    runtime_class_name,
                )
                continue
            runtime_class_name = matched
    if not runtime_class_name:
        return

    # Set the RuntimeClass only if the object exists in the cluster,
    # as naming a missing class makes the kubelet reject the Pod.
    node_api = kubernetes.client.NodeV1Api(client)
    try:
        node_api.read_runtime_class(name=runtime_class_name)
    except kubernetes.client.exceptions.ApiException as e:
        if e.status == 403:
            logger.warning(
                "Workload requests device resources mapped to RuntimeClass '%s', "
                "but the deployer has no permission to read RuntimeClasses, "
                "leaving the Pod runtimeClassName empty",
                runtime_class_name,
            )
            return
        if e.status == 404:
            logger.warning(
                "Workload requests device resources mapped to RuntimeClass '%s', "
                "but that RuntimeClass is not available in the cluster, "
                "leaving the Pod runtimeClassName empty",
                runtime_class_name,
            )
            return
        raise

    logger.info(
        "Setting Pod runtimeClassName to '%s', "
        "resolved from the requested device resources",
        runtime_class_name,
    )
    pod.spec.runtime_class_name = runtime_class_name


def _resolve_privileged(container: Container, kdp: bool) -> bool:
    """
    Resolve whether a container runs privileged.

    Privilege is dropped when the container's devices are handed out by a
    device plugin, which is the case for every device plugin resource family
    and, under the KDP injection policy, for every mapped device request.

    A privileged container receives all device nodes of the host, so it
    enumerates -- and can use -- every accelerator on the node, no matter
    which one the device plugin allocated to it. That silently undoes
    slicing: a workload holding a single MIG device or a single memory slice
    still sees the untouched cards next to it, and a soft-slicing limit
    lands on whichever device comes first instead of the allocated one.

    Args:
        container:
            The container to resolve.
        kdp:
            Whether the KDP injection policy is in effect, resolved once per
            Pod so a workload's containers cannot disagree on it.

    """
    if not container.execution or not container.execution.privileged:
        return False
    requests, _ = container.resolve_resources()
    if not requests:
        return True

    for r_k in requests:
        if r_k in ("cpu", "memory"):
            continue
        if _is_device_plugin_resource(r_k) or (
            kdp
            and (
                r_k
                in envs.GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES
                or r_k == envs.GPUSTACK_RUNTIME_DEPLOY_AUTOMAP_RESOURCE_KEY
            )
        ):
            clogger.info(
                "Dropping privilege of container '%s', "
                "as its device request '%s' is allocated by a device plugin",
                container.name,
                r_k,
            )
            return False
    return True


class KubernetesDeployer(EndoscopicDeployer):
    """
    Deployer implementation for Kubernetes.
    """

    _client: kubernetes.client.ApiClient | None = None
    """
    Client for interacting with the Kubernetes API.
    """
    _node_name: str | None = None
    """
    Name of the node where the deployer is running.
    """
    _image_pull_secrets: dict[WorkloadNamespace, str] | None = None
    """
    Image pull secrets for pulling container images, keyed by the namespace
    they have been applied into: a Pod can only reference a Secret of its own
    namespace, so a deployer serving several workload namespaces holds one
    copy per namespace instead of a single one.
    """
    _mutate_create_pod: (
        Callable[[kubernetes.client.V1Pod], kubernetes.client.V1Pod] | None
    ) = None
    """
    Function to handle mirrored deployment, internal use only.
    """

    @staticmethod
    @ttl_cache(maxsize=1, ttl=60)
    def is_supported() -> bool:
        """
        Check if the deployer is supported in the current environment.

        Returns:
            True if the deployer is supported, False otherwise.

        """
        supported = False
        if envs.GPUSTACK_RUNTIME_DEPLOY.lower() not in ("auto", _NAME):
            return supported

        client = KubernetesDeployer._get_client()
        if client:
            try:
                version_api = kubernetes.client.VersionApi(client)
                version_info = version_api.get_code(_request_timeout=3)
                supported = version_info is not None
                if envs.GPUSTACK_RUNTIME_LOG_EXCEPTION:
                    logger.debug(
                        "Connected to Kubernetes API server: %s",
                        version_info,
                    )
            except kubernetes.client.exceptions.ApiException:
                debug_log_exception(
                    logger,
                    "Failed to connect to Kubernetes API server",
                )
            except urllib3.exceptions.MaxRetryError:
                pass

        return supported

    @staticmethod
    def _get_client(**kwargs) -> kubernetes.client.ApiClient | None:
        """
        Return a Kubernetes API client.

        Args:
            **kwargs:
                Additional arguments to pass to the Kubernetes config loader.

        Returns:
            A Kubernetes API client if the configuration is valid, None otherwise.

        """
        client = None

        try:
            with (
                Path(os.devnull).open("w") as dev_null,
                contextlib.redirect_stdout(dev_null),
                contextlib.redirect_stderr(dev_null),
            ):
                kubernetes.config.load_config(**kwargs)
                client = kubernetes.client.ApiClient()
                client.user_agent = "gpustack/runtime"
        except kubernetes.config.config_exception.ConfigException:
            debug_log_exception(logger, "Failed to get Kubernetes client")

        return client

    @staticmethod
    def _supported(func):
        """
        Decorator to check if Kubernetes is supported in the current environment.
        """

        def wrapper(self, *args, **kwargs):
            if not self.is_supported():
                msg = "Kubernetes is not supported in the current environment."
                raise UnsupportedError(msg)
            return func(self, *args, **kwargs)

        return wrapper

    def _create_ephemeral_configmaps(
        self,
        workload: KubernetesWorkloadPlan,
    ) -> dict[tuple[int, str], str]:
        """
        Create ephemeral files as ConfigMaps in Kubernetes.

        Returns:
            A mapping from (container index, configured path) to actual ConfigMap name.

        Raises:
            OperationError:
                If creating the ConfigMaps fails.

        """
        config_map_name_prefix = workload.name_rfc1123_guard

        ephemeral_filename_mapping: dict[tuple[int, str], str] = {}
        ephemeral_files: list[tuple[str, str]] = []
        for ci, c in enumerate(workload.containers):
            for fi, f in enumerate(c.files or []):
                if f.content is not None:
                    config_map_name = f"{config_map_name_prefix}-{ci}-{fi}"
                    ephemeral_filename_mapping[(ci, f.path)] = config_map_name
                    ephemeral_files.append((config_map_name, f.content))
        if not ephemeral_filename_mapping:
            return ephemeral_filename_mapping

        core_api = kubernetes.client.CoreV1Api(self._client)
        try:
            for config_map_name, content in ephemeral_files:
                config_map = kubernetes.client.V1ConfigMap(
                    metadata=kubernetes.client.V1ObjectMeta(
                        name=config_map_name,
                        namespace=workload.namespace,
                        labels=workload.labels,
                    ),
                    data={"content": content},
                )
                if envs.GPUSTACK_RUNTIME_DEPLOY_PRINT_CONVERSION:
                    clogger.info(
                        f"Creating configmap %s/%s:{os.linesep}%s",
                        workload.namespace,
                        config_map_name,
                        safe_yaml(config_map, indent=2, sort_keys=False),
                    )

                actual_config_map = None
                with contextlib.suppress(kubernetes.client.exceptions.ApiException):
                    actual_config_map = core_api.read_namespaced_config_map(
                        name=config_map_name,
                        namespace=workload.namespace,
                    )
                if not actual_config_map:
                    core_api.create_namespaced_config_map(
                        namespace=workload.namespace,
                        body=config_map,
                    )
                    logger.debug(
                        "Created configmap %s/%s",
                        workload.namespace,
                        config_map_name,
                    )
                elif not equal_config_maps(actual_config_map, config_map):
                    core_api.patch_namespaced_config_map(
                        name=config_map_name,
                        namespace=workload.namespace,
                        body={
                            "data": config_map.data,
                        },
                    )
                    logger.debug(
                        "Updated configmap %s/%s",
                        workload.namespace,
                        config_map_name,
                    )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to create configmap for workload {workload.name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        return ephemeral_filename_mapping

    @staticmethod
    def _append_pod_volumes(
        pod: kubernetes.client.V1Pod,
        workload: KubernetesWorkloadPlan,
        ephemeral_filename_mapping: dict[tuple[int, str], str],
    ):
        """
        Append volumes into the Pod.
        """
        file_volumes: dict[str, kubernetes.client.V1Volume] = {}
        mount_volumes: dict[str, kubernetes.client.V1Volume] = {}

        for ci, c in enumerate(workload.containers):
            for _fi, f in enumerate(c.files or []):
                if f.content is None and f.path:
                    # Host file, bind directly.
                    fp_hash = fnv1a_32_hex(f.path)
                    if fp_hash not in file_volumes:
                        file_volumes[fp_hash] = kubernetes.client.V1Volume(
                            name=f"f-{fp_hash}",
                            host_path=kubernetes.client.V1HostPathVolumeSource(
                                path=f.path,
                                type="File",
                            ),
                        )
                elif f.path:
                    # Ephemeral file, mount from ConfigMap.
                    fn = ephemeral_filename_mapping[(ci, f.path)]
                    if fn not in file_volumes:
                        file_volumes[fn] = kubernetes.client.V1Volume(
                            name=f"f-{fn}",
                            config_map=kubernetes.client.V1ConfigMapVolumeSource(
                                name=fn,
                                items=[
                                    kubernetes.client.V1KeyToPath(
                                        key="content",
                                        path=Path(f.path).name,
                                        mode=f.mode,
                                    ),
                                ],
                            ),
                        )
            for m in c.mounts or []:
                if m.volume is None and m.path:
                    # Host directory, bind directly.
                    mp_hash = fnv1a_32_hex(m.path)
                    if mp_hash not in mount_volumes:
                        mount_volumes[mp_hash] = kubernetes.client.V1Volume(
                            name=f"m-{mp_hash}",
                            host_path=kubernetes.client.V1HostPathVolumeSource(
                                path=m.path,
                                type=(
                                    "DirectoryOrCreate"
                                    if m.mode != ContainerMountModeEnum.ROX
                                    else "Directory"
                                ),
                            ),
                        )
                elif m.path:
                    # Ephemeral volume, mount from emptyDir.
                    if m.volume not in mount_volumes:
                        mount_volumes[m.volume] = kubernetes.client.V1Volume(
                            name=f"m-{m.volume}",
                            empty_dir=kubernetes.client.V1EmptyDirVolumeSource(),
                        )

        if file_volumes or mount_volumes:
            pod.spec.volumes = pod.spec.volumes or []
            if file_volumes:
                pod.spec.volumes.extend(file_volumes.values())
            if mount_volumes:
                pod.spec.volumes.extend(mount_volumes.values())

    @staticmethod
    def _append_container_volume_mounts(
        container: kubernetes.client.V1Container,
        c: Container,
        ci: int,
        ephemeral_filename_mapping: dict[tuple[int, str], str],
    ):
        """
        Append volume mounts into the Container.
        """
        if files := c.files:
            container.volume_mounts = container.volume_mounts or []
            for _fi, f in enumerate(files):
                if f.content is None and f.path:
                    # Host file, mount directly.
                    fp_hash = fnv1a_32_hex(f.path)
                    container.volume_mounts.append(
                        kubernetes.client.V1VolumeMount(
                            name=f"f-{fp_hash}",
                            mount_path=f.path,
                            read_only=(True if f.mode < 0o600 else None),
                        ),
                    )
                elif f.path:
                    # Ephemeral file, mount from ConfigMap.
                    fn = ephemeral_filename_mapping[(ci, f.path)]
                    container.volume_mounts.append(
                        kubernetes.client.V1VolumeMount(
                            name=f"f-{fn}",
                            mount_path=f.path,
                            read_only=(True if f.mode < 0o600 else None),
                            sub_path=Path(f.path).name,
                        ),
                    )

        if mounts := c.mounts:
            container.volume_mounts = container.volume_mounts or []
            for m in mounts:
                if m.volume is None and m.path:
                    # Host directory, mount directly.
                    mp_hash = fnv1a_32_hex(m.path)
                    container.volume_mounts.append(
                        kubernetes.client.V1VolumeMount(
                            name=f"m-{mp_hash}",
                            mount_path=m.path,
                            read_only=(
                                True if m.mode == ContainerMountModeEnum.ROX else None
                            ),
                        ),
                    )
                elif m.volume and m.path:
                    # Ephemeral volume, mount from emptyDir.
                    container.volume_mounts.append(
                        kubernetes.client.V1VolumeMount(
                            name=f"m-{m.volume}",
                            mount_path=m.path,
                            read_only=(
                                True if m.mode == ContainerMountModeEnum.ROX else None
                            ),
                        ),
                    )

    @staticmethod
    def _parameterize_probe(
        check: ContainerCheck,
    ) -> kubernetes.client.V1Probe:
        """
        Parameterize a ContainerCheck into a Kubernetes V1Probe.

        Returns:
            A V1Probe object representing the health check.

        Raises:
            ValueError:
                If the ContainerCheck is invalid.

        """
        probe = kubernetes.client.V1Probe(
            initial_delay_seconds=check.delay,
            period_seconds=check.interval,
            timeout_seconds=check.timeout,
            failure_threshold=check.retries,
            success_threshold=1,
        )

        configured = False
        for attr_k in ["execution", "tcp", "http", "https"]:
            attr_v = getattr(check, attr_k, None)
            if not attr_v:
                continue
            configured = True
            match attr_k:
                case "execution":
                    probe.exec = kubernetes.client.V1ExecAction(
                        command=attr_v.command,
                    )
                case "tcp":
                    probe.tcp_socket = kubernetes.client.V1TCPSocketAction(
                        port=attr_v.port,
                        host=attr_v.host,
                    )
                case "http" | "https":
                    probe.http_get = kubernetes.client.V1HTTPGetAction(
                        path=attr_v.path or "/",
                        port=attr_v.port or 80,
                        host=attr_v.host,
                        scheme="HTTPS" if attr_k == "https" else "HTTP",
                        http_headers=[
                            kubernetes.client.V1HTTPHeader(name=h.name, value=h.value)
                            for h in (attr_v.headers or [])
                        ],
                    )
            break
        if not configured:
            msg = "Invalid health check configuration"
            raise ValueError(msg)

        return probe

    def _probe_node_allocatable(self) -> dict[str, str] | None:
        """
        Read the allocatable resources of the node this deployer targets,
        so the injection policy can tell whether a device plugin runs there.

        Reads through ``list_node`` rather than ``read_node`` so it needs no
        permission beyond the list the deployer already requires to resolve a
        default node name. With no node configured it reads the same first node
        ``_get_default_node_name`` would, and remembers it, so resolving the
        default costs one call rather than two.

        Returns:
            The node's allocatable resources, or None when the node cannot be
            read -- no permission, API error, or no node at all -- so the
            caller can tell "advertises nothing" from "could not look".

        """
        core_api = kubernetes.client.CoreV1Api(self._client)
        try:
            nodes = core_api.list_node(
                field_selector=(
                    f"metadata.name={self._node_name}" if self._node_name else None
                ),
                limit=1,
            )
        except kubernetes.client.exceptions.ApiException as e:
            clogger.warning(
                "Failed to read node allocatable resources"
                "%s, assuming a device plugin is present",
                _detail_api_call_error(e),
            )
            return None

        if not nodes.items:
            return None

        node = nodes.items[0]
        if not self._node_name:
            self._node_name = node.metadata.name
        return node.status.allocatable or {}

    def _resolve_resource_injection_policy(self) -> str:
        """
        Resolve the resource injection policy for this deployer, probing the
        target node when the configured policy is "auto".

        Returns:
            The resource injection policy.

        """
        return get_resource_injection_policy(self._probe_node_allocatable)

    def _get_default_node_name(self) -> str:
        """
        Get the default node name of the cluster.

        Returns:
            The name of the first node in the cluster.

        Raises:
            OperationError:
                If retrieving the node name fails.

        """
        core_api = kubernetes.client.CoreV1Api(self._client)
        try:
            nodes = core_api.list_node(limit=1)
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to get the default node name of the cluster{_detail_api_call_error(e)}"
            raise OperationError(msg) from e
        else:
            if nodes.items:
                return nodes.items[0].metadata.name
            msg = "Failed to get the default node name of the cluster: No nodes found"
            raise OperationError(msg)

    def _apply_default_image_pull_secret(
        self,
        namespace: WorkloadNamespace,
    ) -> str | None:
        """
        Apply the image pull secret of the default container registry into the
        given namespace, if credentials for it are configured.

        Applied per workload namespace rather than once per deployer: a Pod can
        only reference a Secret of its own namespace, so the credentials of the
        default registry need one copy in every namespace deploying a workload
        pulling from it.

        Args:
            namespace:
                The namespace of the workload to pull for.

        Returns:
            The name of the applied image pull secret,
            None if no default registry credentials are configured.

        Raises:
            OperationError:
                If applying the image pull secret fails.

        """
        if not (
            envs.GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME
            and envs.GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD
        ):
            return None

        registry = (
            envs.GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY or "index.docker.io"
        )
        return self._apply_image_pull_secret(
            registry=f"https://{registry}/v1/",
            username=envs.GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME,
            password=envs.GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD,
            namespace=namespace,
        )

    def _apply_image_pull_secret(
        self,
        registry: str,
        username: str,
        password: str,
        namespace: WorkloadNamespace,
    ) -> str:
        """
        Apply image pull secret for pulling container images.

        Args:
            registry:
                The container registry URL.
            username:
                The username for the container registry.
            password:
                The password for the container registry.
            namespace:
                The namespace to apply the image pull secret into,
                which is the namespace of the Pod referencing it: Kubernetes
                resolves an image pull secret in the Pod's own namespace only.

        Returns:
            The name of the created image pull secret.

        Raises:
            OperationError:
                If creating the image pull secret fails.

        """
        core_api = kubernetes.client.CoreV1Api(self._client)

        auth = f"{username}:{password}"
        auth_b64 = base64_encode(auth)

        docker_config = {
            "auths": {
                registry: {
                    "username": username,
                    "password": password,
                    "auth": auth_b64,
                },
            },
        }
        docker_config_json = json.dumps(docker_config)
        docker_config_json_b64 = base64_encode(docker_config_json)

        secret_name = f"gpustack-ips-{fnv1a_64_hex(registry + auth_b64)}"
        secret_namespace = namespace

        # The name hashes the credentials, so a Secret this deployer has
        # already applied under that name in that namespace carries them
        # unchanged: reapplying it would only spend API calls.
        if (self._image_pull_secrets or {}).get(secret_namespace) == secret_name:
            return secret_name

        secret = kubernetes.client.V1Secret(
            metadata=kubernetes.client.V1ObjectMeta(
                name=secret_name,
                namespace=secret_namespace,
            ),
            data={".dockerconfigjson": docker_config_json_b64},
            type="kubernetes.io/dockerconfigjson",
        )

        try:
            actual_secret = None
            with contextlib.suppress(kubernetes.client.exceptions.ApiException):
                actual_secret = core_api.read_namespaced_secret(
                    name=secret_name,
                    namespace=secret_namespace,
                )
            if not actual_secret:
                core_api.create_namespaced_secret(
                    namespace=secret_namespace,
                    body=secret,
                )
                logger.debug(
                    "Created image pull secret %s/%s",
                    secret_namespace,
                    secret_name,
                )
            elif not equal_secrets(actual_secret, secret):
                core_api.patch_namespaced_secret(
                    name=secret_name,
                    namespace=secret_namespace,
                    body={
                        "data": secret.data,
                    },
                )
                logger.debug(
                    "Updated image pull secret %s/%s",
                    secret_namespace,
                    secret_name,
                )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to create image pull secret{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        self._image_pull_secrets = {
            **(self._image_pull_secrets or {}),
            secret_namespace: secret_name,
        }

        return secret_name

    def _create_service(
        self,
        workload: KubernetesWorkloadPlan,
    ):
        """
        Create Kubernetes Service for the workload.

        Returns:
            The created Service object, or None if no ports are defined.

        Raises:
            OperationError:
                If creating the Service fails.

        """
        ports: dict[str, ContainerPort] = {}
        for c in workload.containers:
            if c.profile == ContainerProfileEnum.RUN and c.ports:
                for p in c.ports:
                    pn = f"{p.protocol.lower()}-{p.external or p.internal}".lower()
                    if pn not in ports:
                        ports[pn] = p
        if not ports:
            return None

        service_name = workload.name_rfc1123_guard

        service = kubernetes.client.V1Service(
            metadata=kubernetes.client.V1ObjectMeta(
                name=service_name,
                namespace=workload.namespace,
                labels=workload.labels,
            ),
            spec=kubernetes.client.V1ServiceSpec(
                selector=workload.labels,
                type=workload.service_type,
                ports=[
                    kubernetes.client.V1ServicePort(
                        name=pn,
                        protocol=p.protocol.value,
                        port=p.external or p.internal,
                        target_port=p.internal,
                    )
                    for pn, p in ports.items()
                ],
            ),
        )
        if envs.GPUSTACK_RUNTIME_DEPLOY_PRINT_CONVERSION:
            clogger.info(
                f"Creating service %s/%s:{os.linesep}%s",
                workload.namespace,
                service_name,
                safe_yaml(service, indent=2, sort_keys=False),
            )

        core_api = kubernetes.client.CoreV1Api(self._client)
        try:
            actual_service = None
            with contextlib.suppress(kubernetes.client.exceptions.ApiException):
                actual_service = core_api.read_namespaced_service(
                    name=service_name,
                    namespace=workload.namespace,
                )
            if not actual_service:
                service = core_api.create_namespaced_service(
                    namespace=workload.namespace,
                    body=service,
                )
                logger.debug(
                    "Created service %s/%s",
                    workload.namespace,
                    service_name,
                )
            elif not equal_services(actual_service, service):
                service = core_api.patch_namespaced_service(
                    name=service_name,
                    namespace=workload.namespace,
                    body={
                        "spec": service.spec,
                    },
                )
                logger.debug(
                    "Updated service %s/%s",
                    workload.namespace,
                    service_name,
                )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to create service for workload {workload.name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        return service

    def _create_pod(
        self,
        workload: KubernetesWorkloadPlan,
        ephemeral_filename_mapping: dict[tuple[int, str], str],
    ) -> kubernetes.client.V1Pod:
        """
        Create Kubernetes Pod for the workload.

        Returns:
            The created Pod object.

        Raises:
            ValueError:
                If the workload is invalid.
            OperationError:
                If creating the Pod fails.

        """
        pod_name = workload.name_rfc1123_guard

        pod = kubernetes.client.V1Pod(
            metadata=kubernetes.client.V1ObjectMeta(
                name=pod_name,
                namespace=workload.namespace,
                labels=workload.labels or {},
                # Copied rather than referenced: the conversion stamps its own
                # annotations onto the Pod, e.g. the container names, and must
                # not write them back into the plan it is given.
                annotations=dict(workload.annotations or {}),
            ),
            spec=kubernetes.client.V1PodSpec(
                containers=[],
                host_network=workload.host_network,
                dns_policy=(
                    "ClusterFirstWithHostNet" if workload.host_network else None
                ),
                host_ipc=workload.host_ipc,
                share_process_namespace=workload.pid_shared,
                termination_grace_period_seconds=workload.termination_grace_period_seconds,
                node_name=self._node_name,
                automount_service_account_token=False,
                volumes=(
                    [
                        kubernetes.client.V1Volume(
                            name="dshm",
                            empty_dir=kubernetes.client.V1EmptyDirVolumeSource(
                                medium="Memory",
                                size_limit=workload.shm_size,
                            ),
                        ),
                    ]
                    if not workload.host_ipc and workload.shm_size
                    else None
                ),
                security_context=kubernetes.client.V1PodSecurityContext(
                    run_as_user=workload.run_as_user,
                    run_as_group=workload.run_as_group,
                    fs_group=workload.fs_group,
                    sysctls=(
                        [
                            kubernetes.client.V1Sysctl(name=k, value=v)
                            for k, v in (workload.sysctls or {}).items()
                        ]
                        if workload.sysctls
                        else None
                    ),
                ),
            ),
        )

        # Create dedicated image pull secret if needed, and attach to the Pod.
        # Applied here rather than while preparing the deployer, as the
        # namespace to copy the Secret into is only known once the workload
        # plan has defaulted it, and a Pod resolves an image pull secret in its
        # own namespace only.
        image_pull_secret = self._apply_default_image_pull_secret(workload.namespace)
        for c in workload.containers:
            if c.profile != ContainerProfileEnum.RUN:
                continue
            usernm = passwd = None
            for e in c.envs or []:
                if (
                    e.name
                    == "GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME"
                ):
                    usernm = e.value
                elif (
                    e.name
                    == "GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD"
                ):
                    passwd = e.value
                if usernm and passwd:
                    reg, _, _, _ = parse_image(c.image)
                    reg = reg or "index.docker.io"
                    # Credentials the workload carries itself win over the
                    # configured default ones.
                    image_pull_secret = self._apply_image_pull_secret(
                        registry=f"https://{reg}/v1/",
                        username=usernm,
                        password=passwd,
                        namespace=workload.namespace,
                    )
                    break

        if image_pull_secret:
            pod.spec.image_pull_secrets = [
                kubernetes.client.V1LocalObjectReference(
                    name=image_pull_secret,
                ),
            ]

        # Parameterize volumes
        self._append_pod_volumes(
            pod,
            workload,
            ephemeral_filename_mapping,
        )

        # Resolve the injection policy once per Pod: under "auto" it probes the
        # target node, and every container of the Pod lands on that same node.
        kdp = self._resolve_resource_injection_policy() == "kdp"

        cnt_init, cnt_run = -1, -1
        for ci, c in enumerate(workload.containers):
            # Annotate container info.
            if c.profile == ContainerProfileEnum.INIT:
                cnt_init += 1
                container_annotate_prefix = f"{_LABEL_COMPONENT}-init-{cnt_init}"
            else:
                cnt_run += 1
                container_annotate_prefix = f"{_LABEL_COMPONENT}-run-{cnt_run}"
            pod.metadata.annotations[f"{container_annotate_prefix}-name"] = c.name

            container_name = c.name_rfc1123_guard

            container = kubernetes.client.V1Container(
                name=container_name,
                image=c.image,
                image_pull_policy=c.image_pull_policy,
            )

            if c.profile == ContainerProfileEnum.INIT:
                # Set the restart policy of the INIT container.
                container.restart_policy = c.restart_policy.value
            elif not pod.spec.restart_policy:
                # Set the pod-level restart policy from the first RUN container.
                pod.spec.restart_policy = c.restart_policy.value

            # Parameterize execution
            if c.execution:
                container.working_dir = c.execution.working_dir
                container.command = c.execution.command
                container.args = c.execution.args
                container.security_context = kubernetes.client.V1SecurityContext(
                    run_as_user=c.execution.run_as_user,
                    run_as_group=c.execution.run_as_group,
                    read_only_root_filesystem=c.execution.readonly_rootfs,
                    privileged=_resolve_privileged(c, kdp),
                    capabilities=(
                        kubernetes.client.V1Capabilities(
                            add=c.execution.capabilities.add,
                            drop=c.execution.capabilities.drop,
                        )
                        if c.execution.capabilities
                        else None
                    ),
                )

            # Parameterize environment variables
            container.env = [
                kubernetes.client.V1EnvVar(name=e.name, value=e.value)
                for e in c.envs or []
                if not e.name.startswith("GPUSTACK_RUNTIME_DEPLOY_")
            ]

            # Parameterize resources
            c_requests, c_limits = c.resolve_resources()
            if c_requests:
                fmt = "kdp" if kdp else "plain"

                resources: dict[str, str] = {}
                for r_k, r_v in c_requests.items():
                    if r_k in ("cpu", "memory"):
                        resources[r_k] = str(r_v)
                        continue

                    if (
                        r_k
                        in envs.GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES
                    ):
                        # Set env if resource key is mapped.
                        runtime_envs = [
                            envs.GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES[
                                r_k
                            ],
                        ]
                    elif r_k == envs.GPUSTACK_RUNTIME_DEPLOY_AUTOMAP_RESOURCE_KEY:
                        # Set env if auto-mapping key is matched.
                        runtime_envs = self.get_runtime_envs()
                    else:
                        resources[r_k] = str(r_v)
                        continue

                    privileged = (
                        container.security_context
                        and container.security_context.privileged
                    )
                    resource_values = [x.strip() for x in r_v.split(",")]

                    # Request devices.
                    if r_v == "all":
                        # Configure privileged.
                        if not kdp:
                            container.security_context = (
                                container.security_context
                                or kubernetes.client.V1SecurityContext()
                            )
                            container.security_context.privileged = True
                        # Request all devices.
                        for ren in runtime_envs:
                            r_vs = self.get_runtime_visible_devices(ren, fmt)
                            if kdp:
                                # Request quantity of devices as resources for KDP to schedule on the correct nodes.
                                resources.update(r_vs)
                                continue
                            # Request device via visible devices env.
                            container.env.append(
                                kubernetes.client.V1EnvVar(
                                    name=ren,
                                    value=",".join(r_vs),
                                ),
                            )
                    else:
                        # Request specific devices.
                        for ren in runtime_envs:
                            if kdp:
                                r_vs = self.map_runtime_visible_devices(
                                    ren,
                                    resource_values,
                                    fmt,
                                )
                                # Manually select devices via annotation for KDP.
                                pod.metadata.annotations[
                                    _ANNOTATION_PREFERRED_DEVICE_INDEX
                                ] = ",".join(resource_values)
                                # Request quantity of devices as resources for KDP to schedule on the correct nodes.
                                resources.update(r_vs)
                                continue

                            # Request all devices if privileged,
                            # otherwise, normalize requested devices.
                            if privileged:
                                r_vs = self.get_runtime_visible_devices(ren, fmt)
                            else:
                                r_vs = self.map_runtime_visible_devices(
                                    ren,
                                    resource_values,
                                    fmt,
                                )
                            container.env.append(
                                kubernetes.client.V1EnvVar(
                                    name=ren,
                                    value=",".join(r_vs),
                                ),
                            )

                    # Configure runtime device access environment variables.
                    if not kdp and r_v != "all" and privileged:
                        b_vs = self.map_backend_visible_devices(
                            runtime_envs,
                            resource_values,
                        )
                        container.env.extend(
                            [
                                kubernetes.client.V1EnvVar(
                                    name=be,
                                    value=be_v,
                                )
                                for be, be_v in b_vs.items()
                            ],
                        )

                    # Pin the device ordering whenever the container ends up
                    # seeing more than one device, so its numbering stays
                    # aligned with the detection.
                    # That covers every multi-device request, not only "all":
                    # any container holding several devices numbers them
                    # itself, and a performance-sorted default reshuffles those
                    # ordinals on a heterogeneous host.
                    # Requesting all devices is measured rather than
                    # special-cased -- including under KDP, where the device
                    # plugin allocates every device of the node and the
                    # container still enumerates all of them -- because a
                    # single-device host has nothing to reorder.
                    # A privileged container sees every device of the host
                    # whatever it requested, so it pins regardless.
                    # Never overwrite the ordering declared by the container.
                    if (
                        privileged
                        or self.count_requested_devices(runtime_envs, resource_values)
                        > 1
                    ):
                        declared_envs = {e.name for e in container.env}
                        container.env.extend(
                            [
                                kubernetes.client.V1EnvVar(
                                    name=o_k,
                                    value=o_v,
                                )
                                for o_k, o_v in self.map_visible_devices_ordering(
                                    runtime_envs,
                                ).items()
                                if o_k not in declared_envs
                            ],
                        )

                container.resources = _container_resource_requirements(
                    container_name,
                    resources,
                    c_limits,
                )

            # Parameterize mounts
            self._append_container_volume_mounts(
                container,
                c,
                ci,
                ephemeral_filename_mapping,
            )

            # Parameterize ports
            if c.ports:
                container.ports = [
                    kubernetes.client.V1ContainerPort(
                        container_port=p.internal,
                        host_port=p.external or p.internal
                        if workload.host_network
                        else None,
                        protocol=p.protocol.value,
                    )
                    for p in c.ports
                ]

            # Parameterize health checks
            if c.profile == ContainerProfileEnum.RUN and c.checks:
                # Find the first teardown-enabled check,
                # make it as the liveness probe.
                chk = next(
                    (chk for chk in c.checks if chk.teardown),
                    None,
                )
                if chk:
                    container.liveness_probe = self._parameterize_probe(chk)

                # Find the first non-teardown-enabled check,
                # make it as the readiness probe.
                chk = next(
                    (chk for chk in c.checks if not chk.teardown),
                    None,
                )
                if chk:
                    container.readiness_probe = self._parameterize_probe(chk)

            # Concat shared memory volume if needed.
            if not workload.host_ipc and workload.shm_size:
                if not container.volume_mounts:
                    container.volume_mounts = []
                container.volume_mounts.append(
                    kubernetes.client.V1VolumeMount(
                        name="dshm",
                        mount_path="/dev/shm",  # noqa: S108
                    ),
                )

            # Append the container.
            if c.profile == ContainerProfileEnum.INIT:
                pod.spec.init_containers = pod.spec.init_containers or []
                pod.spec.init_containers.append(container)
            else:
                pod.spec.containers.append(container)

        core_api = kubernetes.client.CoreV1Api(self._client)
        try:
            _apply_instance_type_admission(pod, workload.instance_type)
            _pin_pod_for_kueue(pod, self._node_name)

            _resolve_runtime_class_name(pod, self._client)
            pod = self._mutate_create_pod(pod)
            if envs.GPUSTACK_RUNTIME_DEPLOY_PRINT_CONVERSION:
                clogger.info(
                    f"Creating pod %s/%s:{os.linesep}%s",
                    workload.namespace,
                    pod_name,
                    safe_yaml(pod, indent=2, sort_keys=False),
                )

            actual_pod = None
            with contextlib.suppress(kubernetes.client.exceptions.ApiException):
                actual_pod = core_api.read_namespaced_pod(
                    name=pod_name,
                    namespace=workload.namespace,
                )
            if not actual_pod:
                pod = core_api.create_namespaced_pod(
                    namespace=workload.namespace,
                    body=pod,
                )
                logger.debug(
                    "Created pod %s/%s",
                    workload.namespace,
                    pod_name,
                )
            elif not equal_pods(actual_pod, pod):
                # Delete the existing Pod first, then create a new one.
                with watch(
                    core_api.list_namespaced_pod,
                    resource_version=_get_quorum_read_resource_version(),
                    namespace=workload.namespace,
                ) as es:
                    core_api.delete_namespaced_pod(
                        name=pod_name,
                        namespace=workload.namespace,
                    )
                    for e in es:
                        if (
                            e["type"] == "DELETED"
                            and e["object"].metadata.name == pod_name
                        ):
                            break
                pod = core_api.create_namespaced_pod(
                    namespace=workload.namespace,
                    body=pod,
                )
                logger.debug(
                    "Updated pod %s/%s",
                    workload.namespace,
                    pod_name,
                )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to create pod for workload {workload.name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        return pod

    def __init__(self):
        super().__init__(_NAME)
        self._client = self._get_client()
        self._node_name = envs.GPUSTACK_RUNTIME_KUBERNETES_NODE_NAME
        self._image_pull_secrets = {}
        self._runtime_uuid_values_allowed: bool | None = None

    @property
    def allowed_runtime_uuid_values(self) -> bool:
        # Resolved once per deployer, unlike the per-Pod resolution the
        # creation path wants: this gates how `_prepare` builds the device
        # materials, which are themselves built once, and it is read there once
        # per manufacturer -- so probing on every read would spend one API call
        # per manufacturer to answer a question already settled.
        if self._runtime_uuid_values_allowed is None:
            self._runtime_uuid_values_allowed = (
                self._resolve_resource_injection_policy() != "kdp"
            )
        return self._runtime_uuid_values_allowed

    @property
    def allowed_mig_devices(self) -> bool:
        # On Kubernetes a partitioner (e.g. the GPUStack Operator's device
        # manager) owns MIG: it creates and destroys the instances on demand,
        # so the card is the item to hold on to, and a pod asks for a slice of
        # it by resource rather than by addressing an instance that may not
        # outlive the request.
        return False

    def _prepare_mirrored_deployment(self):
        """
        Prepare for mirrored deployment.

        """
        # Get the first node name of the cluster if not configured.
        # Required list permission on Kubernetes Node resources.
        if not self._node_name:
            self._node_name = self._get_default_node_name()

        # NB(thxCode): The image pull secret of the default registry used to be
        # applied here, but it can only be applied once the namespace to copy
        # it into is known, which the workload plan only defaults later:
        # see `_apply_default_image_pull_secret`.

        # Prepare mirrored deployment if enabled.
        if self._mutate_create_pod:
            return
        self._mutate_create_pod = lambda o: o

        # Retrieve self-pod info.
        try:
            self_pod = self._find_self_pod()
            if not self_pod:
                return
        except kubernetes.client.exceptions.ApiException:
            logger.exception(
                "Mirrored deployment enabled, but failed to get self Pod, skipping",
            )
            return

        self_pod_name = self_pod.metadata.name
        self_pod_namespace = self_pod.metadata.namespace

        # Retrieve self-container
        ## - Get the first Container, or the Container named "default" if exists.
        self_container = next(
            (c for c in self_pod.spec.containers if c.name == "default"),
            None,
        )
        if not self_container:
            self_container = self_pod.spec.containers[0]
            logger.warning(
                "Mirrored deployment enabled, but no Container named 'default' found, using the first Container instead",
            )
        logger.info(
            "Mirrored deployment enabled, using self Container %s of self Pod %s/%s for options mirroring",
            self_container.name,
            self_pod_namespace,
            self_pod_name,
        )

        core_api = kubernetes.client.CoreV1Api(self._client)

        # Namespaces already told what they cannot take from the worker,
        # so redeploying into one does not repeat the same warnings.
        warned_namespaces: set[WorkloadNamespace] = set()

        # Construct mutation function.
        def mutate_create_pod(
            pod: kubernetes.client.V1Pod,
        ) -> kubernetes.client.V1Pod:
            # Preprocess mirrored deployment options.
            # NB(thxCode): Resolved per Pod, not once per deployer: a worker
            # hosts the workloads of several organizations, each in its own
            # namespace, and Kubernetes resolves a Secret, ConfigMap or PVC
            # reference in the referring Pod's namespace alone. Deciding once
            # would settle for every Pod what only holds for the Pods sharing
            # the namespace of the worker itself, and silently strip the rest.
            pod_namespace = pod.metadata.namespace or _default_workload_namespace()
            in_same_namespace = pod_namespace == self_pod_namespace
            # Report a namespace's dropped options once, on its first Pod.
            report_drops = pod_namespace not in warned_namespaces
            warned_namespaces.add(pod_namespace)
            ## - Pod runtime class name
            mirrored_runtime_class_name: str = self_pod.spec.runtime_class_name or ""
            ## - Pod image pull secrets
            mirrored_image_pull_secrets: list[
                kubernetes.client.V1LocalObjectReference
            ] = self_pod.spec.image_pull_secrets or []
            if pod.spec.image_pull_secrets:
                # Use created image pull secret if exists.
                mirrored_image_pull_secrets = []
            elif mirrored_image_pull_secrets and not in_same_namespace:
                # A Pod resolves an image pull secret in its own namespace, so
                # mirroring the worker's names Secrets that do not exist there.
                if report_drops:
                    logger.warning(
                        "Mirrored deployment drops image pull secret(s) %s of self Pod "
                        "for the workloads of namespace %s: "
                        "Kubernetes resolves an image pull secret in the referring Pod's namespace, "
                        "which is not the worker's namespace %s. "
                        "Configure `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME` "
                        "and `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD` "
                        "to have the credentials copied into every workload namespace instead",
                        ", ".join(
                            ips.name for ips in mirrored_image_pull_secrets if ips.name
                        ),
                        pod_namespace,
                        self_pod_namespace,
                    )
                mirrored_image_pull_secrets = []
            ## - Container envs
            mirrored_envs: list[kubernetes.client.V1EnvVar] = [
                # Filter out gpustack-internal envs and cross-namespace secret/envref envs.
                e
                for e in self_container.env or []
                if (
                    not e.name.startswith("GPUSTACK_")
                    and (not e.value_from or in_same_namespace)
                )
            ]
            if not in_same_namespace and report_drops:
                # Name every env left behind: a customer's HF_TOKEN or proxy
                # configuration disappearing without a word is indistinguishable
                # from a workload misbehaving on its own.
                dropped_env_names = [
                    e.name
                    for e in self_container.env or []
                    if not e.name.startswith("GPUSTACK_") and e.value_from
                ]
                if dropped_env_names:
                    logger.warning(
                        "Mirrored deployment drops env(s) %s of self Container %s "
                        "for the workloads of namespace %s: "
                        "Kubernetes resolves a `valueFrom` reference, a Secret or ConfigMap key among them, "
                        "in the referring Pod's namespace, which is not the worker's namespace %s. "
                        "Declare them on the workload itself to have them delivered",
                        ", ".join(dropped_env_names),
                        self_container.name,
                        pod_namespace,
                        self_pod_namespace,
                    )
            igs = envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_ENVIRONMENTS
            if igs:
                mirrored_envs = [
                    # Filter out ignored envs.
                    e
                    for e in mirrored_envs
                    if e.name not in igs
                ]
            ## - Container volume mounts
            mirrored_volume_mounts: list[kubernetes.client.V1VolumeMount] = (
                self_container.volume_mounts or []
            )
            if igs := envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES:
                mirrored_volume_mounts = [
                    # Filter out ignored volume mounts.
                    m
                    for m in mirrored_volume_mounts
                    if m.mount_path not in igs
                ]
            ## - Container volume devices
            mirrored_volume_devices: list[kubernetes.client.V1VolumeDevice] = (
                self_container.volume_devices or []
            )
            if igs := envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES:
                mirrored_volume_devices = [
                    # Filter out ignored volume mounts.
                    d
                    for d in mirrored_volume_devices
                    if d.device_path not in igs
                ]
            ## - Pod volumes
            mirrored_volume_mounts_names = {m.name for m in mirrored_volume_mounts}
            mirrored_volume_devices_names = {d.name for d in mirrored_volume_devices}
            mirrored_volumes: list[kubernetes.client.V1Volume] = []
            dropped_volume_names: list[str] = []
            # Filter out volumes not used by mirrored volume mounts or devices.
            for v in self_pod.spec.volumes or []:
                if (
                    v.name not in mirrored_volume_mounts_names
                    and v.name not in mirrored_volume_devices_names
                ):
                    continue
                # Skip downwardAPI/projected volumes
                if v.downward_api or v.projected:
                    mirrored_volume_mounts_names.discard(v.name)
                    mirrored_volume_devices_names.discard(v.name)
                    continue
                # Skip configMap/secret/PVC volumes if not in same namespace
                if (
                    v.config_map or v.secret or v.persistent_volume_claim
                ) and not in_same_namespace:
                    mirrored_volume_mounts_names.discard(v.name)
                    mirrored_volume_devices_names.discard(v.name)
                    dropped_volume_names.append(v.name)
                    continue
                # Skip PVCs with RWO or RWOP access modes
                if v.persistent_volume_claim:
                    pvc = core_api.read_namespaced_persistent_volume_claim(
                        name=v.persistent_volume_claim.claim_name,
                        namespace=self_pod_namespace,
                    )
                    if "ReadWriteOncePod" in pvc.spec.access_modes or []:
                        mirrored_volume_mounts_names.discard(v.name)
                        mirrored_volume_devices_names.discard(v.name)
                        continue
                mirrored_volumes.append(v)
            if dropped_volume_names and report_drops:
                logger.warning(
                    "Mirrored deployment drops volume(s) %s of self Pod "
                    "for the workloads of namespace %s: "
                    "Kubernetes resolves a ConfigMap, Secret or PersistentVolumeClaim "
                    "in the referring Pod's namespace, "
                    "which is not the worker's namespace %s",
                    ", ".join(dropped_volume_names),
                    pod_namespace,
                    self_pod_namespace,
                )
            ## - Correct Container volume mounts and volume devices without corresponding volumes.
            mirrored_volume_mounts = [
                m
                for m in mirrored_volume_mounts
                if m.name in mirrored_volume_mounts_names
            ]
            mirrored_volume_devices = [
                d
                for d in mirrored_volume_devices
                if d.name in mirrored_volume_devices_names
            ]

            if mirrored_runtime_class_name and not pod.spec.runtime_class_name:
                pod.spec.runtime_class_name = mirrored_runtime_class_name

            if mirrored_image_pull_secrets:
                pod.spec.image_pull_secrets = pod.spec.image_pull_secrets or []
                p_image_pull_secrets_names = {
                    # Map existing image pull secret names.
                    ips.name
                    for ips in pod.spec.image_pull_secrets
                }
                for ips in mirrored_image_pull_secrets:
                    if ips.name not in p_image_pull_secrets_names:
                        pod.spec.image_pull_secrets.append(ips)
                        p_image_pull_secrets_names.add(ips.name)

            if mirrored_envs:
                for ci in pod.spec.containers:
                    c_env_names = {ei.name for ei in ci.env or []}
                    for ei in mirrored_envs:
                        if ei.name not in c_env_names:
                            ci.env.append(ei)

            if mirrored_volume_mounts or mirrored_volume_devices:
                for ci in pod.spec.containers:
                    ci.volume_mounts = ci.volume_mounts or []
                    c_volume_mount_names = {
                        # Map existing volume mount names.
                        mi.name
                        for mi in ci.volume_mounts
                    }
                    c_volume_mount_paths = {
                        # Map existing volume mount paths.
                        mi.mount_path
                        for mi in ci.volume_mounts
                    }
                    # Append volume mounts if not exists.
                    for mi in mirrored_volume_mounts:
                        if (
                            mi.name not in c_volume_mount_names
                            and mi.mount_path not in c_volume_mount_paths
                        ):
                            ci.volume_mounts.append(mi)
                            c_volume_mount_names.add(mi.name)
                            c_volume_mount_paths.add(mi.mount_path)
                    ci.volume_devices = ci.volume_devices or []
                    # Append volume devices if not exists.
                    c_volume_device_names = {
                        # Map existing volume device names.
                        di.name
                        for di in ci.volume_devices
                    }
                    c_volume_device_paths = {
                        # Map existing volume device paths.
                        di.device_path
                        for di in ci.volume_devices
                    }
                    for di in mirrored_volume_devices:
                        if (
                            di.name not in c_volume_device_names
                            and di.device_path not in c_volume_device_paths
                        ):
                            ci.volume_devices.append(di)
                            c_volume_device_names.add(di.name)
                            c_volume_device_paths.add(di.device_path)

            if mirrored_volumes:
                pod.spec.volumes = pod.spec.volumes or []
                p_volume_names = {
                    # Map existing volume names.
                    vi.name
                    for vi in pod.spec.volumes
                }
                for vi in mirrored_volumes:
                    if vi.name not in p_volume_names:
                        pod.spec.volumes.append(vi)
                        p_volume_names.add(vi.name)

            return pod

        self._mutate_create_pod = mutate_create_pod

    def _find_self_pod(self) -> kubernetes.client.V1Pod | None:
        """
        Find the self Pod in the cluster.

        Returns:
            The self Pod object if found, None otherwise.

        Raises:
            If failed to find itself.

        """
        if not envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT:
            logger.debug("Mirrored deployment disabled")
            return None

        # Get Pod name or hostname.
        self_pod_name = envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME
        if not self_pod_name:
            msg = "Please use env `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME` to specify the exact Pod name"
            raise kubernetes.client.exceptions.ApiException(
                status=404,
                reason=msg,
            )

        # Get Pod namespace, default to "default" if not found.
        try:
            self_pod_namespace_f = Path(
                "/var/run/secrets/kubernetes.io/serviceaccount/namespace",
            )
            self_pod_namespace = self_pod_namespace_f.read_text(
                encoding="utf-8",
            ).strip()
        except (FileNotFoundError, OSError):
            self_pod_namespace = "default"
            logger.warning(
                "Mirrored deployment enabled, but no Pod namespace found, using 'default' instead",
            )

        core_api = kubernetes.client.CoreV1Api(self._client)

        return core_api.read_namespaced_pod(
            name=self_pod_name,
            namespace=self_pod_namespace,
        )

    @_supported
    def _create(self, workload: WorkloadPlan):
        """
        Deploy a Kubernetes workload.

        Args:
            workload:
                The workload to deploy.

        Raises:
            TypeError:
                If the Docker workload type is invalid.
            ValueError:
                If the Kubernetes workload fails to validate.
            UnsupportedError:
                If Kubernetes is not supported in the current environment.
            OperationError:
                If the Kubernetes workload fails to deploy.

        """
        if not isinstance(workload, KubernetesWorkloadPlan | WorkloadPlan):
            msg = f"Invalid workload plan type: {type(workload)}"
            raise TypeError(msg)

        self._prepare_mirrored_deployment()

        if isinstance(workload, WorkloadPlan):
            workload = KubernetesWorkloadPlan(**workload.__dict__)
        workload.validate_and_default()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Creating workload:\n%s", workload.to_yaml())

        # Create ephemeral file if needed,
        # (container index, configured path): <actual ConfigMap name>
        ephemeral_filename_mapping: dict[tuple[int, str], str] = (
            self._create_ephemeral_configmaps(workload)
        )

        # Create Service if needed.
        self._create_service(workload)

        # Create Pod.
        self._create_pod(
            workload,
            ephemeral_filename_mapping,
        )

    def _workload_namespace(
        self,
        namespace: WorkloadNamespace | None = None,
        name: WorkloadName | None = None,
    ) -> WorkloadNamespace:
        """
        Resolve the namespace a workload lives in.

        Args:
            namespace:
                The namespace the caller declared, if any.
            name:
                The name of the workload, which lets a caller declaring no
                namespace search the workload namespaces for it.

        Returns:
            The namespace holding the workload: the declared one, the one the
            search finds it in, or the configured default.

        """
        if namespace:
            return namespace
        if name:
            found = self._search_workload_namespace(name)
            if found:
                return found
        return _default_workload_namespace()

    def _search_workload_namespace(
        self,
        name: WorkloadName,
    ) -> WorkloadNamespace | None:
        """
        Search the workload namespaces for the one holding the given workload,
        which is what an operation carrying a name only has to do: the
        namespace belongs to the workload, so the name alone cannot tell it.

        Args:
            name:
                The name of the workload.

        Returns:
            The namespace holding the workload, None if the search finds none
            or cannot be made at all.

        """
        try:
            k_pods = self._list_workload_pods(
                label_selector=f"{_LABEL_WORKLOAD}={name}",
                resource_version=_get_quorum_read_resource_version(),
            )
        except kubernetes.client.exceptions.ApiException:
            # The search only narrows a namespace the caller left open, so a
            # failed one degrades to the configured default rather than
            # failing the operation asking for it.
            debug_log_exception(
                logger,
                "Failed to search the namespace of workload %s",
                name,
            )
            return None

        namespaces = sorted({k_pod.metadata.namespace for k_pod in k_pods})
        if not namespaces:
            return None
        if len(namespaces) > 1:
            # Two organizations may name a workload alike, so pick
            # deterministically and say which one was picked.
            logger.warning(
                "Workload %s found in multiple namespaces %s, using %s",
                name,
                namespaces,
                namespaces[0],
            )
        return namespaces[0]

    def _list_workload_pods(
        self,
        namespace: WorkloadNamespace | None = None,
        **list_options,
    ) -> list[kubernetes.client.V1Pod]:
        """
        List the workload Pods matching the given options.

        A caller declaring a namespace reads that namespace only, which is the
        single read it has always been. A caller declaring none cannot know
        which organization namespace holds the workload, so it reads every
        namespace at once and keeps the Pods of the workload namespaces; a
        cluster not letting the deployer read Pods cluster-wide degrades to the
        configured default namespace, the only namespace it used to read.

        Args:
            namespace:
                The namespace to read, every workload namespace if None.
            **list_options:
                Options to pass to the list call, e.g. the label selector.

        Returns:
            The Pods found.

        Raises:
            kubernetes.client.exceptions.ApiException:
                If the Pods fail to list.

        """
        core_api = kubernetes.client.CoreV1Api(self._client)

        if namespace:
            k_pods = core_api.list_namespaced_pod(
                namespace=namespace,
                **list_options,
            )
            return k_pods.items or []

        try:
            k_pods = core_api.list_pod_for_all_namespaces(**list_options)
        except kubernetes.client.exceptions.ApiException:
            fallback_namespace = _default_workload_namespace()
            debug_log_exception(
                logger,
                "Failed to list pods across namespaces, falling back to namespace %s",
                fallback_namespace,
            )
            k_pods = core_api.list_namespaced_pod(
                namespace=fallback_namespace,
                **list_options,
            )
            return k_pods.items or []

        return [
            k_pod
            for k_pod in k_pods.items or []
            if _is_workload_namespace(k_pod.metadata.namespace)
        ]

    @_supported
    def _get(
        self,
        name: WorkloadName,
        namespace: WorkloadNamespace | None = None,
    ) -> WorkloadStatus | None:
        """
        Get the status of a Kubernetes workload.

        Args:
            name:
                The name of the workload.
            namespace:
                The namespace of the workload.

        Returns:
            The status if found, None otherwise.

        Raises:
            UnsupportedError:
                If Kubernetes is not supported in the current environment.
            OperationError:
                If the Kubernetes workload fails to get.

        """
        list_options = {
            "label_selector": f"{_LABEL_WORKLOAD}={name}",
            "resource_version": _get_quorum_read_resource_version(),
        }

        core_api = kubernetes.client.CoreV1Api(self._client)

        try:
            # A caller declaring no namespace reads the workload namespaces,
            # so it finds the workload wherever the organization owning it
            # deploys instead of in the deployer's own namespace only.
            k_pods = self._list_workload_pods(
                namespace=namespace,
                **list_options,
            )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to get deployment of workload {name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        if len(k_pods) > 1:
            namespaced_names = [
                f"{d.metadata.namespace}/{d.metadata.name}" for d in k_pods
            ]
            logger.warning(
                "Multiple pods found for workload %s: %s",
                name,
                namespaced_names,
            )

        if not k_pods:
            return None

        k_pod = k_pods[0]
        return KubernetesWorkloadStatus(
            name=name,
            k_pod=k_pod,
            core_api=core_api,
        )

    @_supported
    def _delete(
        self,
        name: WorkloadName,
        namespace: WorkloadNamespace | None = None,
        grace_period_seconds: int | None = None,
    ) -> WorkloadStatus | None:
        """
        Delete a Kubernetes workload.

        Args:
            name:
                The name of the workload.
            namespace:
                The namespace of the workload.
            grace_period_seconds:
                Duration in seconds the workload needs to terminate gracefully,
                which overrides the one declared by the workload plan.

        Returns:
            The status if found, None otherwise.

        Raises:
            UnsupportedError:
                If Kubernetes is not supported in the current environment.
            OperationError:
                If the Kubernetes workload fails to delete.

        """
        # Check if the workload exists.
        workload = self.get(name=name, namespace=namespace)
        if not workload:
            return None

        # Delete from the namespace holding the workload, which the lookup
        # above reports: deleting from the deployer's own namespace instead
        # leaves the Pod running, and with it the devices it holds. The lookup
        # has already searched the workload namespaces, so there is nothing
        # left to resolve beyond the fallback every operation shares.
        namespace = (
            getattr(workload, "namespace", None)
            or namespace
            or _default_workload_namespace()
        )

        resource_version = _get_quorum_read_resource_version()
        label_selector = f"{_LABEL_WORKLOAD}={name}"
        propagation_policy = envs.GPUSTACK_RUNTIME_KUBERNETES_DELETE_PROPAGATION_POLICY

        core_api = kubernetes.client.CoreV1Api(self._client)

        # Remove all Pods with the workload label.
        try:
            core_api.delete_collection_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector,
                propagation_policy=propagation_policy,
                grace_period_seconds=grace_period_seconds,
            )
        except kubernetes.client.exceptions.ApiException as e:
            if e.status != 405:
                msg = f"Failed to delete pod of workload {name}{_detail_api_call_error(e)}"
                raise OperationError(msg) from e
            try:
                pods = core_api.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=label_selector,
                    resource_version=resource_version,
                )
                for pod in pods.items or []:
                    core_api.delete_namespaced_pod(
                        name=pod.metadata.name,
                        namespace=namespace,
                        propagation_policy=propagation_policy,
                        grace_period_seconds=grace_period_seconds,
                    )
            except Exception as e2:
                msg = f"Failed to delete pod of workload {name}{_detail_api_call_error(e2)}"
                raise OperationError(msg) from e2
        # NB(thxCode): Deleting a workload always fails with an OperationError,
        # so the transport errors underneath the API errors do not escape either.
        except Exception as e:
            msg = f"Failed to delete pod of workload {name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        # Remove all Services with the workload label.
        try:
            core_api.delete_collection_namespaced_service(
                namespace=namespace,
                label_selector=label_selector,
                propagation_policy=propagation_policy,
            )
        except kubernetes.client.exceptions.ApiException as e:
            # If method not allowed(405),
            # list services with the label and delete them one by one.
            if e.status != 405:
                msg = f"Failed to delete service of workload {name}{_detail_api_call_error(e)}"
                raise OperationError(msg) from e
            try:
                services = core_api.list_namespaced_service(
                    namespace=namespace,
                    label_selector=label_selector,
                    resource_version=resource_version,
                )
                for svc in services.items or []:
                    core_api.delete_namespaced_service(
                        name=svc.metadata.name,
                        namespace=namespace,
                        propagation_policy=propagation_policy,
                    )
            except Exception as e2:
                msg = f"Failed to delete service of workload {name}{_detail_api_call_error(e2)}"
                raise OperationError(msg) from e2
        # NB(thxCode): Deleting a workload always fails with an OperationError,
        # so the transport errors underneath the API errors do not escape either.
        except Exception as e:
            msg = f"Failed to delete service of workload {name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        # Remove all ConfigMaps with the workload label.
        try:
            core_api.delete_collection_namespaced_config_map(
                namespace=namespace,
                label_selector=label_selector,
                propagation_policy=propagation_policy,
            )
        except kubernetes.client.exceptions.ApiException as e:
            if e.status != 405:
                msg = f"Failed to delete configmap of workload {name}{_detail_api_call_error(e)}"
                raise OperationError(msg) from e
            try:
                configmaps = core_api.list_namespaced_config_map(
                    namespace=namespace,
                    label_selector=label_selector,
                    resource_version=resource_version,
                )
                for cm in configmaps.items or []:
                    core_api.delete_namespaced_config_map(
                        name=cm.metadata.name,
                        namespace=namespace,
                        propagation_policy=propagation_policy,
                    )
            except Exception as e2:
                msg = f"Failed to delete configmap of workload {name}{_detail_api_call_error(e2)}"
                raise OperationError(msg) from e2
        # NB(thxCode): Deleting a workload always fails with an OperationError,
        # so the transport errors underneath the API errors do not escape either.
        except Exception as e:
            msg = f"Failed to delete configmap of workload {name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        return workload

    @_supported
    def _list(
        self,
        namespace: WorkloadNamespace | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[WorkloadStatus]:
        """
        List all Kubernetes workloads.

        Args:
            namespace:
                The namespace of the workloads.
            labels:
                Labels to filter the workloads.

        Returns:
            A list of workload statuses.

        Raises:
            UnsupportedError:
                If Kubernetes is not supported in the current environment.
            OperationError:
                If the Kubernetes workloads fail to list.

        """
        list_options = {
            "label_selector": ",".join(
                [
                    *[
                        f"{k}={v}"
                        for k, v in (labels or {}).items()
                        if k != _LABEL_WORKLOAD
                    ],
                    _LABEL_WORKLOAD,
                ],
            ),
            "resource_version": _get_quorum_read_resource_version(),
        }

        core_api = kubernetes.client.CoreV1Api(self._client)

        try:
            # A caller declaring no namespace sweeps the workload namespaces:
            # a worker hosts the workloads of several organizations, so the
            # deployer's own namespace lists none of them.
            k_pods = self._list_workload_pods(
                namespace=namespace,
                **list_options,
            )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to list workloads' deployments{_detail_api_call_error(e)}"
            raise OperationError(msg) from e

        return [
            KubernetesWorkloadStatus(
                name=k_pod.metadata.labels[_LABEL_WORKLOAD],
                k_pod=k_pod,
                core_api=core_api,
            )
            for k_pod in k_pods
            if (
                k_pod.metadata.labels
                and _LABEL_WORKLOAD in k_pod.metadata.labels
                and k_pod.spec.node_name == self._node_name
            )
        ]

    @_supported
    def _logs(
        self,
        name: WorkloadName,
        namespace: WorkloadNamespace | None = None,
        token: WorkloadOperationToken | None = None,
        timestamps: bool = False,
        tail: int | None = None,
        since: int | None = None,
        follow: bool = False,
    ) -> Generator[bytes | str, None, None] | bytes | str:
        """
        Get logs of a Kubernetes workload or a specific container.

        Args:
            name:
                The name of the workload.
            namespace:
                The namespace of the workload.
            token:
                The operation token to identify the container.
            timestamps:
                Whether to include timestamps in the logs.
            tail:
                Number of lines from the end of the logs to retrieve.
            since:
                Only return logs newer than a relative duration in seconds.
            follow:
                Whether to stream the logs.

        Returns:
            The logs as a byte string, a string or a generator yielding byte strings or strings if follow is True.

        Raises:
            UnsupportedError:
                If Kubernetes is not supported in the current environment.
            OperationError:
                If the Kubernetes workload logs fail to retrieve.

        """
        # The namespace is resolved by the lookup, which searches the workload
        # namespaces when the caller declares none, and the Pod it finds
        # carries the namespace the log read below needs.
        workload = self.get(name=name, namespace=namespace)
        if not workload:
            msg = f"Workload {name} not found"
            raise OperationError(msg)

        k_pod = getattr(workload, "_k_pod", kubernetes.client.V1Pod())
        container = next(
            (
                c
                for ci, c in enumerate(k_pod.spec.containers)
                if (c.name == token if token else ci == 0)
            ),
            None,
        )
        if not container:
            msg = f"Loggable container of workload {name} not found"
            if token:
                msg += f" with token {token}"
            raise OperationError(msg)

        logs_options = {
            "timestamps": timestamps,
            "tail_lines": tail if tail >= 0 else None,
            "since_seconds": since,
            "follow": follow,
            "_preload_content": not follow,
        }

        core_api = kubernetes.client.CoreV1Api(self._client)

        try:
            output = core_api.read_namespaced_pod_log(
                namespace=k_pod.metadata.namespace,
                name=k_pod.metadata.name,
                container=container.name,
                **logs_options,
            )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to fetch logs for container {container.name} of workload {name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e
        else:
            return output

    @_supported
    def _exec(
        self,
        name: WorkloadName,
        namespace: WorkloadNamespace | None = None,
        token: WorkloadOperationToken | None = None,
        detach: bool = True,
        command: list[str] | None = None,
        args: list[str] | None = None,
    ) -> WorkloadExecStream | bytes | str:
        """
        Execute a command in a Kubernetes workload or a specific container.

        Args:
            name:
                The name of the workload.
            namespace:
                The namespace of the workload.
            token:
                The operation token to identify the container.
            detach:
                Whether to run the command in detached mode.
            command:
                The command to execute.
            args:
                The arguments to pass to the command.

        Returns:
            If detach is False, return a WorkloadExecStream.
            otherwise, return the output of the command as a byte string or string.

        Raises:
            UnsupportedError:
                If Kubernetes is not supported in the current environment.
            OperationError:
                If the Kubernetes workload exec fails.

        """
        # The namespace is resolved by the lookup, which searches the workload
        # namespaces when the caller declares none, and the Pod it finds
        # carries the namespace the exec below needs.
        workload = self.get(name=name, namespace=namespace)
        if not workload:
            msg = f"Workload {name} not found"
            raise OperationError(msg)

        k_pod = getattr(workload, "_k_pod", kubernetes.client.V1Pod())
        container = next(
            (
                c
                for ci, c in enumerate(k_pod.spec.containers)
                if (c.name == token if token else ci == 0)
            ),
            None,
        )
        if not container:
            msg = f"Executable container of workload {name} not found"
            if token:
                msg += f" with token {token}"
            raise OperationError(msg)

        attach = not detach or not command
        exec_options = {
            "stdout": True,
            "stderr": True,
            "stdin": attach,
            "tty": attach,
            "command": [*command, *(args or [])] if command else ["/bin/sh"],
            "_preload_content": not attach,
        }

        core_api = kubernetes.client.CoreV1Api(self._client)

        try:
            result = kubernetes.stream.stream(
                core_api.connect_get_namespaced_pod_exec,
                namespace=k_pod.metadata.namespace,
                name=k_pod.metadata.name,
                container=container.name,
                **exec_options,
            )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to exec command in container {container.name} of workload {name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e
        else:
            if not attach:
                return result
            return KubernetesWorkloadExecStream(result)

    @_supported
    def _inspect(
        self,
        name: WorkloadName,
        namespace: WorkloadNamespace | None = None,
    ) -> str | None:
        """
        Inspect a Kubernetes workload.

        Args:
            name:
                The name of the workload.
            namespace:
                The namespace of the workload.

        Returns:
            The inspection result as a YAML string if found, None otherwise.

        Raises:
            UnsupportedError:
                If Kubernetes is not supported in the current environment.
            OperationError:
                If the Kubernetes workload fails to inspect.

        """
        workload = self._get(name=name, namespace=namespace)
        if not workload:
            return None

        k_pod = getattr(workload, "_k_pod", None)
        if not k_pod:
            return None

        # Remove managed fields to reduce output size
        k_pod.metadata.managed_fields = None
        # Mask sensitive environment variables
        for c in k_pod.spec.containers:
            for env in c.env or []:
                if sensitive_env_var(env.name):
                    env.value = "******"

        return safe_yaml(k_pod, indent=2, sort_keys=False)

    def _find_self_pod_for_endoscopy(self) -> kubernetes.client.V1Pod:
        """
        Find the self Pod for endoscopy.
        Only works in mirrored deployment mode.

        Returns:
            The self Pod object.

        Raises:
            UnsupportedError:
                If endoscopy is not supported in the current environment.

        """
        try:
            self_pod = self._find_self_pod()
        except kubernetes.client.exceptions.ApiException as e:
            msg = "Endoscopy is not supported in the current environment: Mirrored deployment enabled, but failed to get self Pod"
            raise UnsupportedError(msg) from e
        except Exception as e:
            msg = "Endoscopy is not supported in the current environment: Failed to get self Pod"
            raise UnsupportedError(msg) from e

        if not self_pod:
            msg = "Endoscopy is not supported in the current environment: Mirrored deployment disabled"
            raise UnsupportedError(msg)
        return self_pod

    def _endoscopic_logs(
        self,
        timestamps: bool = False,
        tail: int | None = None,
        since: int | None = None,
        follow: bool = False,
    ) -> Generator[bytes | str, None, None] | bytes | str:
        """
        Get the logs of the deployer itself.
        Only works in mirrored deployment mode.

        Args:
            timestamps:
                Show timestamps in the logs.
            tail:
                Number of lines to show from the end of the logs.
            since:
                Show logs since the given epoch in seconds.
            follow:
                Whether to follow the logs.

        Returns:
            The logs as a byte string or a generator yielding byte strings if follow is True.

        Raises:
            UnsupportedError:
                If endoscopy is not supported in the current environment.
            OperationError:
                If the deployer fails to get logs.

        """
        self_pod = self._find_self_pod_for_endoscopy()

        logs_options = {
            "timestamps": timestamps,
            "tail_lines": tail if tail >= 0 else None,
            "since_seconds": since,
            "follow": follow,
            "_preload_content": not follow,
        }

        self_pod_name = self_pod.metadata.name
        self_pod_namespace = self_pod.metadata.namespace

        core_api = kubernetes.client.CoreV1Api(self._client)

        try:
            output = core_api.read_namespaced_pod_log(
                namespace=self_pod_namespace,
                name=self_pod_name,
                **logs_options,
            )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to fetch logs for self Pod {self_pod_namespace}/{self_pod_name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e
        else:
            return output

    def _endoscopic_exec(
        self,
        detach: bool = True,
        command: list[str] | None = None,
        args: list[str] | None = None,
    ) -> WorkloadExecStream | bytes | str:
        """
        Execute a command in the deployer itself.
        Only works in mirrored deployment mode.

        Args:
            detach:
                Whether to detach from the command.
            command:
                The command to execute.
                If not specified, use /bin/sh and implicitly attach.
            args:
                The arguments to pass to the command.

        Returns:
            If detach is False, return a WorkloadExecStream.
            otherwise, return the output of the command as a byte string or string.

        Raises:
            UnsupportedError:
                If endoscopy is not supported in the current environment.
            OperationError:
                If the deployer fails to execute the command.

        """
        self_pod = self._find_self_pod_for_endoscopy()

        attach = not detach or not command
        exec_options = {
            "stdout": True,
            "stderr": True,
            "stdin": attach,
            "tty": attach,
            "command": [*command, *(args or [])] if command else ["/bin/sh"],
            "_preload_content": not attach,
        }

        self_pod_name = self_pod.metadata.name
        self_pod_namespace = self_pod.metadata.namespace

        core_api = kubernetes.client.CoreV1Api(self._client)

        try:
            result = kubernetes.stream.stream(
                core_api.connect_get_namespaced_pod_exec,
                namespace=self_pod_namespace,
                name=self_pod_name,
                **exec_options,
            )
        except kubernetes.client.exceptions.ApiException as e:
            msg = f"Failed to exec command in self Pod {self_pod_namespace}/{self_pod_name}{_detail_api_call_error(e)}"
            raise OperationError(msg) from e
        else:
            if not attach:
                return result
            return KubernetesWorkloadExecStream(result)

    def _endoscopic_inspect(self) -> str:
        """
        Inspect the deployer itself.
        Only works in mirrored deployment mode.

        Returns:
            The inspection result.

        Raises:
            UnsupportedError:
                If endoscopy is not supported in the current environment.
            OperationError:
                If the deployer fails to execute the command.

        """
        self_pod = self._find_self_pod_for_endoscopy()

        # Remove managed fields to reduce output size
        self_pod.metadata.managed_fields = None
        # Mask sensitive environment variables
        for c in self_pod.spec.containers:
            for env in c.env or []:
                if sensitive_env_var(env.name):
                    env.value = "******"

        return safe_yaml(self_pod, indent=2, sort_keys=False)


def equal_config_maps(
    a: kubernetes.client.V1ConfigMap,
    b: kubernetes.client.V1ConfigMap,
) -> bool:
    """
    Compare two Kubernetes ConfigMap specs for equality, ignoring certain fields.

    Args:
        a:
            The first ConfigMap spec.
        b:
            The second ConfigMap spec.

    Returns:
        True if the ConfigMap specs are equal, False otherwise.

    """
    return (a.data or {}) == (b.data or {})


def equal_secrets(
    a: kubernetes.client.V1Secret,
    b: kubernetes.client.V1Secret,
) -> bool:
    """
    Compare two Kubernetes Secret specs for equality, ignoring certain fields.

    Args:
        a:
            The first Secret spec.
        b:
            The second Secret spec.

    Returns:
        True if the Secret specs are equal, False otherwise.

    """
    return (a.data or {}) == (b.data or {})


def equal_services(
    a: kubernetes.client.V1Service,
    b: kubernetes.client.V1Service,
) -> bool:
    """
    Compare two Kubernetes Service specs for equality, ignoring certain fields.

    Args:
        a:
            The first Service spec.
        b:
            The second Service spec.

    Returns:
        True if the Service specs are equal, False otherwise.

    """
    aspec = a.spec
    bspec = b.spec
    if (aspec.selector or {}) != (bspec.selector or {}):
        return False
    if aspec.type != bspec.type:
        return False
    return (aspec.ports or []) == (bspec.ports or [])


_RE_QUANTITY_NUMBER = re.compile(
    r"^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$",
)
"""
Regex for the number a Kubernetes resource quantity carries,
which accepts ASCII digits only.
"""

_QUANTITY_SUFFIXES: dict[str, Decimal] = {
    "Ki": Decimal(2) ** 10,
    "Mi": Decimal(2) ** 20,
    "Gi": Decimal(2) ** 30,
    "Ti": Decimal(2) ** 40,
    "Pi": Decimal(2) ** 50,
    "Ei": Decimal(2) ** 60,
    "n": Decimal(10) ** -9,
    "u": Decimal(10) ** -6,
    "m": Decimal(10) ** -3,
    "k": Decimal(10) ** 3,
    "M": Decimal(10) ** 6,
    "G": Decimal(10) ** 9,
    "T": Decimal(10) ** 12,
    "P": Decimal(10) ** 15,
    "E": Decimal(10) ** 18,
}
"""
Multipliers of the suffixes a Kubernetes resource quantity may carry.
"""


def _parse_quantity(value: float | str) -> Decimal | str:
    """
    Parse a Kubernetes resource quantity into the number it denotes,
    so the spellings the API server rewrites, e.g. "1.5Gi" into "1536Mi"
    or 0.5 into "500m", still compare equal to what was requested.

    Args:
        value:
            The resource quantity to parse.

    Returns:
        The number the quantity denotes,
        or the value itself if it does not spell one.

    """
    text = str(value)

    number, multiplier = text, Decimal(1)
    for suffix, suffix_multiplier in _QUANTITY_SUFFIXES.items():
        if text.endswith(suffix):
            number, multiplier = text[: -len(suffix)], suffix_multiplier
            break

    # NB(thxCode): Decimal is more permissive than Kubernetes, which accepts
    # ASCII digits only, so an underscore separated or full-width spelling
    # would otherwise read as the plain one
    # and let an invalid declaration pass as an unchanged one.
    if not _RE_QUANTITY_NUMBER.match(number):
        return text

    return Decimal(number) * multiplier


def _container_resources(
    container: kubernetes.client.V1Container,
) -> dict:
    """
    Return the resources the given Container spec requests as numbers,
    dropping the empty entries the API server fills in,
    which an unset resources declaration does not carry.

    """
    if container.resources is None:
        return {}
    return {
        kind: (
            {k: _parse_quantity(v) for k, v in values.items()}
            if isinstance(values, dict)
            else values
        )
        for kind, values in container.resources.to_dict().items()
        if values
    }


def equal_containers(
    a: kubernetes.client.V1Container,
    b: kubernetes.client.V1Container,
) -> bool:
    """
    Compare two Kubernetes Container specs for equality, ignoring certain fields.

    Args:
        a:
            The first Container spec.
        b:
            The second Container spec.

    Returns:
        True if the Container specs are equal, False otherwise.

    """
    if a.name != b.name:
        return False
    if a.restart_policy != b.restart_policy:
        return False
    if a.image != b.image:
        return False
    if (a.command or []) != (b.command or []):
        return False
    if (a.args or []) != (b.args or []):
        return False
    if (a.working_dir or "") != (b.working_dir or ""):
        return False
    if (a.ports or []) != (b.ports or []):
        return False
    if _container_resources(a) != _container_resources(b):
        return False
    if (a.volume_mounts or []) != (b.volume_mounts or []):
        return False
    if (a.volume_devices or []) != (b.volume_devices or []):
        return False
    if (a.liveness_probe or {}) != (b.liveness_probe or {}):
        return False
    if (a.readiness_probe or {}) != (b.readiness_probe or {}):
        return False
    if (a.security_context or {}) != (b.security_context or {}):
        return False
    if len(a.env or []) != len(b.env or []):
        return False
    aenv = {e.name: e.value for e in a.env or []}
    benv = {e.name: e.value for e in b.env or []}
    return all(not (k not in benv or benv[k] != v) for k, v in aenv.items())


_MANAGED_POD_LABELS = frozenset(
    {
        _LABEL_WORKLOAD,
        _LABEL_KUEUE_QUEUE_NAME,
        _LABEL_KUEUE_POD_GROUP_NAME,
    },
)
"""
Labels this deployer declares on a Pod and therefore compares.
"""

_MANAGED_POD_ANNOTATIONS = frozenset(
    {
        _ANNOTATION_KUEUE_POD_GROUP_TOTAL_COUNT,
        _ANNOTATION_PREFERRED_DEVICE_INDEX,
    },
)
"""
Annotations this deployer declares on a Pod and therefore compares.
"""

_MANAGED_POD_METADATA_PREFIXES = (f"{envs.GPUSTACK_RUNTIME_DEPLOY_LABEL_PREFIX}/",)
"""
Prefixes of the label and annotation keys this deployer owns outright, e.g. the
per-container name annotations, whose exact keys depend on the workload.
"""

_FOREIGN_MANAGED_POD_METADATA_PREFIXES = ("kueue.x-k8s.io/",)
"""
Prefixes of the managed keys another controller also writes, which are
therefore compared one way only, see `_equal_managed_metadata`.
"""


def _managed_metadata(
    entries: dict[str, str] | None,
    managed_keys: frozenset[str],
) -> dict[str, str]:
    """
    Return the entries of the given labels or annotations this deployer
    declares, which are the ones it may have to change.

    Comparing the whole metadata instead would churn on every reconcile: the
    API server, the admission webhooks and the controllers taking over the Pod
    all stamp their own annotations on it, e.g. the CNI's or Kueue's, and none
    of those says anything about the workload being deployed.

    Args:
        entries:
            The labels or annotations to filter.
        managed_keys:
            The exact keys this deployer declares.

    Returns:
        The managed entries.

    """
    return {
        k: v
        for k, v in (entries or {}).items()
        if k in managed_keys or k.startswith(_MANAGED_POD_METADATA_PREFIXES)
    }


def _equal_managed_metadata(
    actual: dict[str, str] | None,
    desired: dict[str, str] | None,
    managed_keys: frozenset[str],
) -> bool:
    """
    Compare the managed entries of two label or annotation sets.

    The keys this deployer owns outright compare both ways: nothing else
    writes them, so a difference in either direction is one this deployer
    made, e.g. renaming a container.

    The managed keys another controller also writes -- the Kueue ones, which
    an admission webhook may default or add -- compare one way only: the
    desired value must be there, and an entry only the actual Pod carries is
    left alone. Comparing those both ways would have the deployer recreate a
    Pod on every reconcile as soon as the webhook adds one of them.

    Args:
        actual:
            The labels or annotations of the actual Pod.
        desired:
            The labels or annotations of the desired Pod.
        managed_keys:
            The exact keys this deployer declares.

    Returns:
        True if the managed entries are equal, False otherwise.

    """
    actual_managed = _managed_metadata(actual, managed_keys)
    desired_managed = _managed_metadata(desired, managed_keys)

    for k, v in desired_managed.items():
        if actual_managed.get(k) != v:
            return False

    return all(
        k in desired_managed
        for k in actual_managed
        if not k.startswith(_FOREIGN_MANAGED_POD_METADATA_PREFIXES)
    )


def _equal_pod_node_selectors(
    actual: kubernetes.client.V1PodSpec,
    desired: kubernetes.client.V1PodSpec,
) -> bool:
    """
    Compare the node selectors of two Pod specs, one way only: every selector
    the desired spec declares must be there, and a selector only the actual Pod
    carries is left alone, as admission adds its own -- Kueue copies the node
    labels of the ResourceFlavor it admitted the workload on into the Pod.

    Args:
        actual:
            The spec of the actual Pod.
        desired:
            The spec of the desired Pod.

    Returns:
        True if the actual Pod carries the declared selectors, False otherwise.

    """
    actual_selector = actual.node_selector or {}
    return all(
        actual_selector.get(k) == v for k, v in (desired.node_selector or {}).items()
    )


def _pod_termination_grace_period_seconds(
    spec: kubernetes.client.V1PodSpec,
) -> int:
    """
    Return the termination grace period the given Pod spec settles on,
    which is the API server default when the spec leaves it unset.

    """
    if spec.termination_grace_period_seconds is None:
        return DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS
    return spec.termination_grace_period_seconds


def equal_pods(
    a: kubernetes.client.V1Pod,
    b: kubernetes.client.V1Pod,
) -> bool:
    """
    Compare two Kubernetes Pods for equality, ignoring certain fields.

    Args:
        a:
            The actual Pod, i.e. the one the cluster holds.
        b:
            The desired Pod, i.e. the one this deployer declares.

    Returns:
        True if the Pods are equal, False otherwise.

    """
    # The metadata this deployer declares decides as much as the spec does: the
    # gang markers admitting a workload as a whole live in the labels and the
    # annotations, so a Pod matching on the spec alone would keep running
    # ungrouped, and changing the size of a group would never reach it.
    if not _equal_managed_metadata(
        a.metadata.labels if a.metadata else None,
        b.metadata.labels if b.metadata else None,
        _MANAGED_POD_LABELS,
    ):
        return False
    if not _equal_managed_metadata(
        a.metadata.annotations if a.metadata else None,
        b.metadata.annotations if b.metadata else None,
        _MANAGED_POD_ANNOTATIONS,
    ):
        return False

    aspec = a.spec
    bspec = b.spec
    if not _equal_pod_node_selectors(aspec, bspec):
        return False
    if len(aspec.init_containers or []) != len(bspec.init_containers or []):
        return False
    for ac, bc in zip(
        aspec.init_containers or [],
        bspec.init_containers or [],
        strict=False,
    ):
        if not equal_containers(ac, bc):
            return False
    if aspec.runtime_class_name != bspec.runtime_class_name:
        return False
    # NB(thxCode): The API server drops the disabled toggles instead of
    # echoing them back, so compare what they settle on, not what they carry.
    if bool(aspec.host_network) != bool(bspec.host_network):
        return False
    if bool(aspec.host_ipc) != bool(bspec.host_ipc):
        return False
    if bool(aspec.share_process_namespace) != bool(bspec.share_process_namespace):
        return False
    if (aspec.restart_policy or "Always") != (bspec.restart_policy or "Always"):
        return False
    if _pod_termination_grace_period_seconds(
        aspec,
    ) != _pod_termination_grace_period_seconds(bspec):
        return False
    if aspec.node_name and bspec.node_name and aspec.node_name != bspec.node_name:
        return False
    if (aspec.volumes or []) != (bspec.volumes or []):
        return False
    if (aspec.security_context or {}) != (bspec.security_context or {}):
        return False
    if len(aspec.containers or []) != len(bspec.containers or []):
        return False
    for ac, bc in zip(
        aspec.containers or [],
        bspec.containers or [],
        strict=False,
    ):
        if not equal_containers(ac, bc):
            return False

    return True


_WATCH_TIMEOUT_SECONDS = 300
"""
Duration in seconds a watch streams events for at most, after which the API
server closes it and the loop reading it ends.
Generous enough for a Pod to drain within the grace periods a workload
declares, which is what the watches here wait for.
"""


@contextlib.contextmanager
def watch(func, *args, **kwargs):
    # Bound every watch: an unbounded one turns a dropped connection or an
    # event that never comes into an operation waiting forever, and the loop
    # reading the stream only ends when the stream does.
    kwargs.setdefault("timeout_seconds", _WATCH_TIMEOUT_SECONDS)

    w: kubernetes.watch.Watch | None = None
    try:
        w = kubernetes.watch.Watch()
        yield w.stream(func, *args, **kwargs)
    finally:
        if w:
            w.stop()


class KubernetesWorkloadExecStream(WorkloadExecStream):
    """
    A WorkloadExecStream implementation for Kubernetes exec streams.
    """

    _ws: kubernetes.stream.ws_client.WSClient | None = None

    def __init__(self, ws: kubernetes.stream.ws_client.WSClient):
        super().__init__()
        self._ws = ws
        self._ws.run_forever(timeout=1)
        self._ws.write_stdin(b" \b")

    @property
    def closed(self) -> bool:
        return not (self._ws and self._ws.is_open())

    def fileno(self) -> int:
        return self._ws.sock.fileno()

    def recv(self, size=-1) -> bytes | None:
        return self.read(size)

    def send(self, data: bytes) -> int:
        return self.write(data)

    def read(self, *_) -> bytes | None:
        if self.closed:
            return None
        self._ws.update(timeout=1)
        return self._ws.read_all().encode("utf-8", errors="replace")

    def write(self, data: bytes) -> int:
        if self.closed:
            return 0
        data_len = len(data)
        self._ws.write_stdin(data.decode("utf-8", errors="replace"))
        return data_len

    def close(self):
        if not self.closed:
            return
        self._ws.close()


def _detail_api_call_error(err: Exception) -> str:
    """
    Explain a Kubernetes API error in a concise way,
    if the envs.GPUSTACK_RUNTIME_DEPLOY_API_CALL_ERROR_DETAIL is enabled.

    Args:
        err:
            The Kubernetes API error, or the transport error underneath it.

    Returns:
        A concise explanation of the error.

    """
    if not envs.GPUSTACK_RUNTIME_DEPLOY_API_CALL_ERROR_DETAIL:
        return ""

    if not isinstance(err, kubernetes.client.exceptions.ApiException):
        return f": {err}"

    msg = ": Kubernetes Error"
    if err.reason:
        msg += f": {err.reason}"
    elif err.body:
        msg += f": {err.body}"
    else:
        msg += f": status code {err.status}"

    return msg


def _get_quorum_read_resource_version() -> str | None:
    """
    Get the resource version for quorum read based on environment settings.

    """
    return None if envs.GPUSTACK_RUNTIME_KUBERNETES_QUORUM_READ else "0"
