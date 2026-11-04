"""Real local vector index through VikingFS, SearchService and context assembly."""

from openviking.core.namespace import classify_uri
from openviking.models.embedder.base import EmbedResult
from openviking.server.identity import RequestContext, Role
from openviking.service.search_service import SearchService
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.viking_fs import VikingFS
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig


class FixedEmbedder:
    def prepare_embedding_input(self, text):
        return text

    async def embed_async(self, text, *, is_query=False):
        return EmbedResult(dense_vector=[1.0, 0.0, 0.0, 0.0])


async def test_real_local_index_filters_category_and_type_before_candidate_cut(tmp_path):
    ctx = RequestContext(UserIdentifier("quota-test", "alice"), Role.USER)
    store = VikingVectorIndexBackend(
        VectorDBBackendConfig(path=str(tmp_path), name="quota_context", dimension=4)
    )
    try:
        assert await store.create_collection(
            "quota_context", CollectionSchemas.context_collection("quota_context", 4)
        )
        records = [
            (f"viking://user/alice/memories/events/{i}.md", "memory", [1.0, 0.0]) for i in range(50)
        ]
        # Another 50 records contain the desired segment under an unrelated category.
        records += [
            (f"viking://user/alice/memories/custom/preferences/{i}.md", "memory", [1.0, 0.0])
            for i in range(50)
        ]
        expected = {
            "preferences": "viking://user/alice/memories/preferences/answer.md",
            "resources": "viking://user/alice/resources/experiences/guide.md",
            "skills": "viking://user/alice/skills/deploy/SKILL.md",
        }
        for category, uri in expected.items():
            records.append(
                (
                    uri,
                    {"resources": "resource", "skills": "skill"}.get(category, "memory"),
                    [0.5, 0.866],
                )
            )
        for i, (uri, context_type, vector) in enumerate(records):
            await store.upsert(
                {
                    "id": str(i),
                    "uri": uri,
                    "context_type": context_type,
                    "level": 2,
                    "abstract": f"answer {i}",
                    "vector": vector + [0.0, 0.0],
                    "account_id": ctx.account_id,
                    "owner_user_id": "alice",
                    "active_count": 0,
                },
                ctx=ctx,
            )
        # The branch uses recursive retrieval: index the directory ancestors
        # just as semantic processing does, without mocking directory traversal.
        directories = {}
        for uri, context_type, _ in records:
            path = classify_uri(uri)
            for length in range(path.content_index + 1, len(path.parts)):
                directories["viking://" + "/".join(path.parts[:length])] = context_type
        for i, (uri, context_type) in enumerate(directories.items()):
            await store.upsert(
                {
                    "id": f"dir-{i}",
                    "uri": uri,
                    "context_type": context_type,
                    "level": 0,
                    "abstract": f"directory {uri}",
                    "vector": [0.0, 1.0, 0.0, 0.0],
                    "account_id": ctx.account_id,
                    "owner_user_id": "alice",
                    "active_count": 0,
                },
                ctx=ctx,
            )
        # Demonstrate that a broad top-20 query really does lose all target rows.
        broad = await store.search_in_tenant(
            ctx,
            query_vector=[1.0, 0.0, 0.0, 0.0],
            target_directories=["viking://user/alice"],
            limit=20,
        )
        assert len(broad) == 20
        assert not ({r["uri"] for r in broad} & set(expected.values()))

        fs = VikingFS(agfs=None, query_embedder=FixedEmbedder(), vector_store=store)
        service = SearchService(fs)
        for category, uri in expected.items():
            result = await service.search(
                "q",
                ctx=ctx,
                mode="context",
                quotas={category: 1},
                detail="abstract",
                score_threshold=0,
            )
            assert not result["stats"]["retrieval_errors"], result
            assert [e["uri"] for e in result["entries"]] == [uri], result
            assert result["entries"][0]["category"] == category
            assert not result["stats"]["retrieval_errors"]
    finally:
        await store.drop_collection()
        await store.close()
