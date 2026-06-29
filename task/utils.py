from task.user_client import User

def join_context(context: list[User]) -> str:
    # TODO:
    # You cannot pass raw JSON with user data to LLM (" sign), collect it in just simple string or markdown.
    # You need to collect it in such way:
    # User:
    #   name: John
    #   surname: Doe
    #   ...
    content = ""
    small_spacing = " " * 2
    medium_spacing = " " * 4

    for user in context:
        user_content = "User:\n"

        user_credit_card = user.credit_card.model_dump() if user.credit_card else {}
        user_address = user.address.model_dump() if user.address else {}

        for key, value in user.model_dump(exclude={"credit_card", "address"}).items():
            item = f"{small_spacing} {key}: {value}"
            user_content += item + "\n"

        address_header = "Address:\n"
        user_content += address_header
        for key, value in user_address.items():
            item = f"{medium_spacing} {key}: {value}"
            user_content += item + "\n"

        credit_card_header = "Credit Card:\n"
        user_content += credit_card_header
        for key, value in user_credit_card.items():
            item = f"{medium_spacing} {key}: {value}"
            user_content += item + "\n"

        content += "\n" + user_content + "\n\n"

    return content