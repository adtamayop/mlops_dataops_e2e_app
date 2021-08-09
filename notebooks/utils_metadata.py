import pandas as pd
from ml_metadata.proto import metadata_store_pb2


def display_artifacts(store, artifacts, base_dir):
    # Helper function to render dataframes for the input artifacts
    table = {"artifact id": [], "type": [], "uri": []}
    for a in artifacts:
        table["artifact id"].append(a.id)
        artifact_type = store.get_artifact_types_by_id([a.type_id])[0]
        table["type"].append(artifact_type.name)
        table["uri"].append(a.uri.replace(base_dir, "./"))
    return pd.DataFrame(data=table)


def display_properties(store, node):
    # Helper function to render dataframes for artifact and execution properties
    table = {"property": [], "value": []}
    for k, v in node.properties.items():
        table["property"].append(k)
        table["value"].append(
            v.string_value if v.HasField("string_value") else v.int_value
        )
    for k, v in node.custom_properties.items():
        table["property"].append(k)
        table["value"].append(
            v.string_value if v.HasField("string_value") else v.int_value
        )
    return pd.DataFrame(data=table)


def display_types(types):
    # Helper function to render dataframes for the artifact and execution types
    table = {"id": [], "name": []}
    for a_type in types:
        table["id"].append(a_type.id)
        table["name"].append(a_type.name)
    return pd.DataFrame(data=table)


def get_one_hop_parent_artifacts(store, artifacts):
    # Get a list of artifacts within a 1-hop of the artifacts of interest
    artifact_ids = [artifact.id for artifact in artifacts]
    executions_ids = set(
        event.execution_id
        for event in store.get_events_by_artifact_ids(artifact_ids)
        if event.type == metadata_store_pb2.Event.OUTPUT
    )
    artifacts_ids = set(
        event.artifact_id
        for event in store.get_events_by_execution_ids(executions_ids)
        if event.type == metadata_store_pb2.Event.INPUT
    )
    return [artifact for artifact in store.get_artifacts_by_id(artifacts_ids)]


def find_producer_execution(store, artifact):
    executions_ids = set(
        event.execution_id
        for event in store.get_events_by_artifact_ids([artifact.id])
        if event.type == metadata_store_pb2.Event.OUTPUT
    )
    return store.get_executions_by_id(executions_ids)[0]
