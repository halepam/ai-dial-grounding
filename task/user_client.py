from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel
import requests

from task._constants import USER_SERVICE_ENDPOINT


class Address(BaseModel):
    country: str
    city: str
    street: str
    flat_house: str


class CreditCard(BaseModel):
    num: str
    cvv: str
    exp_date: str


class User(BaseModel):
    id: int
    name: str
    surname: str
    email: str
    phone: str
    date_of_birth: date
    address: Address
    gender: str
    company: Optional[str] = None
    salary: Optional[float] = None
    about_me: str
    credit_card: CreditCard
    created_at: datetime


class UserClient:
    def get_all_users(self) -> list[User]:
        headers = {"Content-Type": "application/json"}

        response = requests.get(
            url=USER_SERVICE_ENDPOINT + "/v1/users", headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Get {len(data)} users successfully")
            return [User(**u) for u in data]

        raise Exception(f"HTTP {response.status_code}: {response.text}")

    async def get_user(self, id: int) -> User:
        headers = {"Content-Type": "application/json"}

        response = requests.get(
            url=f"{USER_SERVICE_ENDPOINT}/v1/users/{id}", headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            return User(**data)

        raise Exception(f"HTTP {response.status_code}: {response.text}")

    def search_users(
        self,
        name: Optional[str] = None,
        surname: Optional[str] = None,
        email: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> list[User]:
        headers = {"Content-Type": "application/json"}

        # Only include parameters that are not None
        params = {}
        if name:
            params["name"] = name
        if surname:
            params["surname"] = surname
        if email:
            params["email"] = email
        if gender:
            params["gender"] = gender

        response = requests.get(
            url=USER_SERVICE_ENDPOINT + "/v1/users/search",
            headers=headers,
            params=params,
        )

        if response.status_code == 200:
            data = response.json()
            print(f"Get {len(data)} users successfully")
            return [User(**u) for u in data]

        raise Exception(f"HTTP {response.status_code}: {response.text}")

    def health(self):
        headers = {"Content-Type": "application/json"}

        response = requests.get(url=USER_SERVICE_ENDPOINT + "/health", headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data

        raise Exception(f"HTTP {response.status_code}: {response.text}")
