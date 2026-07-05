import asyncio
from typing import Optional, Any

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, RootModel
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient, User
from task.utils import join_context_user_id_and_aboutme

from pathlib import Path


# TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

SYSTEM_PROMPT = """
You are a helpful assistant that provides accurate answers based on provided context. You will be given a user question and relevant RAG context retrieved from a database. 
Your task is to analyze the context and provide a comprehensive but accurate answer to the user's question.

## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
"""

# TODO:
# Should consist retrieved context and user question
USER_PROMPT = """
## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""

NEE_PROMPT = """
## INSTRUCTIONS:
- Extract all hobbies from the `RAG CONTEXT` that are relevant to the `USER QUESTION`.
- For each hobby, provide a list of user IDs that are associated with that hobby.
- Ensure that the output is in valid JSON format, with hobbies as keys and lists of user IDs as values.
- If no relevant hobbies are found, return an empty JSON object: `{{}}`
## RAG CONTEXT:
{context}
## USER QUESTION:
{query}
## OUTPUT FORMAT:
{{
  "hobby1": [user_id1, user_id2, ...],
  "hobby2": [user_id3, user_id4, ...],
}}
"""


class LLMAnswer(RootModel[dict[str, list[int]]]):
    """
    Top-level mapping: hobby -> list of user IDs.
    """

    pass


def format_user_document(user: User) -> str:
    # TODO:
    # Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    return join_context_user_id_and_aboutme([user])


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None
        self.collection_name = "users_collection"
        self.persistent_path = "./user_wizard_db"
        self.user_client = UserClient()

    async def __aenter__(self):
        print("🔎 Loading all users...")

        user_client = UserClient()
        users = user_client.get_all_users()

        documents = [
            Document(
                page_content=format_user_document(user),
                id=user.id,
                metadata={"id": user.id},
            )
            for user in users
        ]

        self.vectorstore = await self._init_vectorstore_with_batching(documents)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _read_vectorstore(self) -> Optional[Chroma]:
        if self.vectorstore is None:
            if Path.exists(Path("./user_wizard_db")):
                return Chroma(
                    collection_name=self.collection_name,
                    embeddings=self.embeddings,
                    persist_directory=self.persistent_path,
                )
        return self.vectorstore

    async def _init_vectorstore_with_batching(
        self, documents: list[Document], batch_size: int = 100
    ) -> Chroma:
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        # Initialize vectorstore with the first batch
        final_vector_store = await Chroma.afrom_documents(
            batches[0],
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persistent_path,
        )

        # Add remaining batches concurrently into the same vectorstore
        tasks = [final_vector_store.aadd_documents(batch) for batch in batches[1:]]

        await asyncio.gather(*tasks)

        return final_vector_store

    def _users_to_documents(self, users: list[User]) -> list[Document]:
        documents = [
            Document(
                page_content=format_user_document(user),
                id=user.id,
                metadata={"id": user.id},
            )
            for user in users
        ]
        return documents

    async def sync_store(self):
        """
        On every user request we will update vectorstore on the retrieval step:
         - we will remove deleted users
         - and add new users
         - it will also resolve the issue with consistency within these 2 services and will reduce costs (we don't need on each user request to load vectorstore from scratch and pay for it).
        """
        vector_store = self._read_vectorstore()
        users = self.user_client.get_all_users()
        documents = self._users_to_documents(users)

        # Get existing IDs in the vectorstore
        ids: list[dict[str, int]] = vector_store.get().get("metadatas", [])
        # print(f"Vector Store Dict: {ids}")
        existing_ids = {str(obj["id"]) for obj in ids}
        updated_ids = {str(doc.id) for doc in documents}

        ids_to_remove = existing_ids.difference(updated_ids)
        # print(f"Ids missing: {ids_to_remove}")
        if len(ids_to_remove) > 0:
            vector_store.delete(ids=list(ids_to_remove))

        tasks = []
        for doc in documents:
            if doc.id not in existing_ids:
                task = vector_store.aadd_documents([doc])
                tasks.append(task)

        await asyncio.gather(*tasks)

        self.vectorstore = vector_store

    def retrieve_context(self, user_question: str, k=10, score=0.1) -> str:
        """
        Retrieve context from vectorstore based on user question.
        """
        vector_store = self._read_vectorstore()
        if vector_store is None:
            raise ValueError("Vectorstore is not initialized.")

        # Perform similarity search
        results = vector_store.similarity_search_with_relevance_scores(
            query=user_question, k=k, score_threshold=score
        )

        # Collect context from retrieved documents
        context_parts = [doc[0].page_content for doc in results]
        retrieved_context = "\n\n".join(context_parts)

        return retrieved_context

    def _sanitize_user_ids(self, user_ids: list[int]) -> list[int]:
        """
        Sanitize user IDs by removing duplicates and ensuring they are valid integers.
        """
        uniques = set()

        def _is_integer(value: Any) -> bool:
            try:
                if isinstance(value, bool):
                    return False
                return isinstance(value, int)
            except (ValueError, TypeError):
                return False

        def _is_positive(value: Any) -> bool:
            try:
                return int(value) > 0
            except (ValueError, TypeError):
                return False

        for uid in user_ids:
            if uid in uniques:
                continue
            if _is_integer(uid) and _is_positive(uid):
                uniques.add(int(uid))

        return list(uniques)

    def _output_grounding(self, answer: dict[str, list[int]]) -> dict[str, list[User]]:
        """
        Ground the output by fetching full user information based on the extracted user IDs.
        """
        grounded_output = {}
        cached_calls: dict[int, User] = {}
        for hobby, user_ids in answer.items():
            sanitized_user_ids = self._sanitize_user_ids(user_ids)
            grounded_users: list[User] = []
            user_id_set = set(sanitized_user_ids)
            for user_id in sanitized_user_ids:
                try:
                    if user_id in cached_calls:
                        user = cached_calls[user_id]
                    else:
                        user = self.user_client.get_user(user_id)
                        if user and user.id == user_id:
                            cached_calls[user_id] = user

                    if user and user.id == user_id:
                        grounded_users.append(user)
                    else:
                        print(f"User ID {user_id} not found or mismatched.")
                except Exception as e:
                    print(f"Error fetching user {user_id}: {e}")
            not_found_ids = user_id_set - {
                user.id for user in grounded_users
            }  # This will give us the set of user IDs that were not found
            if not_found_ids:
                print(f"User IDs not found for hobby '{hobby}': {not_found_ids}")
            grounded_output[hobby] = grounded_users
        return grounded_output

    def augment_prompt(self, user_question: str, retrieved_context: str) -> str:
        """
        Augment the user prompt with retrieved context.
        """
        augmented_prompt = NEE_PROMPT.format(
            context=retrieved_context, query=user_question
        )
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> dict[str, list[User]]:
        """
        Generate answer using the LLM client based on the augmented prompt.
        """
        parser = PydanticOutputParser(pydantic_object=LLMAnswer)
        system_prompt = SystemMessagePromptTemplate.from_template(
            template=SYSTEM_PROMPT
        )
        user_prompt = HumanMessage(content=augmented_prompt)
        prompt = ChatPromptTemplate.from_messages(
            messages=[system_prompt, user_prompt]
        ).partial(format_instructions=parser.get_format_instructions())
        parsed_result = (prompt | self.llm_client | parser).invoke({})
        print(f"Results: {parsed_result.root}")
        return self._output_grounding(parsed_result.root)


async def main():
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small-1",
        api_key=SecretStr(API_KEY),
        dimensions=384,
        azure_endpoint=DIAL_URL,
    )
    llm_client = AzureChatOpenAI(
        api_key=SecretStr(API_KEY),
        deployment_name="gpt-4o",
        azure_endpoint=DIAL_URL,
        api_version="",
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Vectorstore is ready.")

        print("Query samples:")
        print(" - I need user ids that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ["quit", "exit"]:
                break
            # TODO:
            # 1. Retrieve context
            await (
                rag.sync_store()
            )  # Sync vectorstore with the latest users before retrieval

            retrieved_context = rag.retrieve_context(user_question)
            # print(f"retrieved_context: {retrieved_context}")
            # 2. Make augmentation
            augmented_prompt = rag.augment_prompt(user_question, retrieved_context)
            # 3. Generate answer and print it
            answer = rag.generate_answer(augmented_prompt)
            # rag.generate_answer(augmented_prompt)

            print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
