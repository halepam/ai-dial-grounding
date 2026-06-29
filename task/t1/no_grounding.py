import asyncio

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient, User
from task.utils import join_context

# TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

# {
#     "id": 9,
#     "name": "Chad",
#     "surname": "Simmons",
#     "email": "chad.simmons576@outlook.com",
#     "phone": "+1-374-683-4169",
#     "date_of_birth": "1954-09-21",
#     "address": {
#       "country": "Monaco",
#       "city": "North Rickeymouth",
#       "street": "04045 John Meadows",
#       "flat_house": "Unit 20"
#     },
#     "gender": "male",
#     "company": "Scott, Cox and Woods",
#     "salary": 37763.77,
#     "about_me": "I find joy in gaming and am constantly inspired by politics. My empathetic spirit helps me appreciate rock climbing and remain committed to master new skills.",
#     "credit_card": {
#       "num": "4218-3343-3207-9497",
#       "cvv": "875",
#       "exp_date": "09/2030"
#     },
#     "created_at": "2026-06-28T05:51:06.133555"
#   },

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            "total_tokens": self.total_tokens,
            "batch_count": len(self.batch_tokens),
            "batch_tokens": self.batch_tokens,
        }


# TODO:
# 1. Create AzureChatOpenAI client
#    hint: api_version set as empty string if you gen an error that indicated that api_version cannot be None
# 2. Create TokenTracker
client = AzureChatOpenAI(
    deployment_name="gpt-4o",
    api_key=SecretStr(API_KEY),
    api_version="",
    azure_endpoint=DIAL_URL,
)
token_tracker = TokenTracker()


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # TODO:
    # 1. Create messages array with system prompt and user message
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=user_message)

    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    ai_message = await client.ainvoke(input=[system_message, human_message])

    # 3. Get usage (hint, usage can be found in response metadata (its dict) and has name 'token_usage', that is also
    #    dict and there you need to get 'total_tokens')
    token_usage = ai_message.response_metadata.get("token_usage", {}).get(
        "total_tokens", 0
    )
    # 4. Add tokens to `token_tracker`

    token_tracker.add_tokens(token_usage)
    # 5. Print response content and `total_tokens`
    print(f"Response: {ai_message.content}")
    print(f"Total tokens used: {token_tracker.get_summary()['total_tokens']}")
    # 5. return response content
    return ai_message.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        # TODO:
        # 1. Get all users (use UserClient)
        user_client = UserClient()
        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        users: list[User] = user_client.get_all_users()
        batch_size = 100
        user_batches = [
            users[i : i + batch_size] for i in range(0, len(users), batch_size)
        ]
        # 3. Prepare tasks for async run of response generation for users batches:
        #       - create array tasks
        #       - iterate through `user_batches` and call `generate_response` with these params:
        #           - BATCH_SYSTEM_PROMPT (system prompt)
        #           - User prompt, you need to format USER_PROMPT with context from user batch and user question
        generation_tasks = []
        for user_batch in user_batches:
            user_batch = [User.model_validate(user) for user in user_batch]
            context = join_context(user_batch)
            task = generate_response(
                system_prompt=BATCH_SYSTEM_PROMPT,
                user_message=USER_PROMPT.format(context=context, query=user_question),
            )
            generation_tasks.append(task)
        # 4. Run task asynchronously, use method `gather` form `asyncio`
        results = await asyncio.gather(*generation_tasks)
        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        filtered_results = [r for r in results if r != "NO_MATCHES_FOUND"]
        # 5. If results after filtration are present:
        #       - combine filtered results with "\n\n" spliterator
        #       - generate response with such params:
        #           - FINAL_SYSTEM_PROMPT (system prompt)
        #           - User prompt: you need to make augmentation of retrieved result and user question
        if len(filtered_results) > 0:
            combined_results = "\n\n".join(filtered_results)
            system_message = SystemMessage(content=FINAL_SYSTEM_PROMPT)
            human_message = HumanMessage(
                content=USER_PROMPT.format(
                    context=combined_results, query=user_question
                )
            )
            await client.ainvoke(input=[system_message, human_message])
        else:
            # 6. Otherwise prin the info that `No users found matching`
            print("No users found matching the search criteria.")
        # 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)
        print(f"Total tokens used: {token_tracker.get_summary()['total_tokens']}")


if __name__ == "__main__":
    # context = join_context(
    #     [
    #         User(
    #             id=9,
    #             name="Chad",
    #             surname="Simmons",
    #             email="chad.simmons@example.com",
    #             phone="+1-374-683-4169",
    #             date_of_birth=date(1954, 9, 21),
    #             gender="male",
    #             company="Scott, Cox and Woods",
    #             salary=37763.77,
    #             about_me="I find joy in gaming and am constantly inspired by politics. My empathetic spirit helps me appreciate rock climbing and remain committed to master new skills.",
    #             created_at=datetime(2026, 6, 28, 5, 51, 6, 133555),
    #             address=Address(
    #                 country="Monaco",
    #                 city="North Rickeymouth",
    #                 street="04045 John Meadows",
    #                 flat_house="Unit 20",
    #             ),
    #             credit_card=CreditCard(
    #                 num="4218-3343-3207-9497", cvv="875", exp_date="09/2030"
    #             ),
    #         ),
    #         User(
    #             id=10,
    #             name="Jane",
    #             surname="Doe",
    #             email="jane.doe@example.com",
    #             phone="+1-374-683-4170",
    #             date_of_birth=date(1985, 5, 15),
    #             about_me="I am passionate about traveling and exploring new cultures. I enjoy hiking and photography, and I am always looking for new adventures.",
    #             salary=50000.00,
    #             company="Doe Enterprises",
    #             created_at=datetime(2026, 7, 1, 10, 30, 0, 0),
    #             gender="female",
    #             address=Address(
    #                 country="Monaco",
    #                 city="North Rickeymouth",
    #                 street="04046 John Meadows",
    #                 flat_house="Unit 21",
    #             ),
    #             credit_card=CreditCard(
    #                 num="4218-3343-3207-9498", cvv="876", exp_date="10/2030"
    #             ),
    #         ),
    #     ]
    # )
    # print(f"\ncontext: \n{context}")
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation
