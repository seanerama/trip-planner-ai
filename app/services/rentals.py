import os
import httpx
from typing import Optional
from bs4 import BeautifulSoup
from pydantic import BaseModel


class RentalListing(BaseModel):
    source: str
    name: str
    location: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[float] = None
    url: Optional[str] = None
    image_url: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    guests: Optional[int] = None


class SearchParams(BaseModel):
    location: str
    checkin: Optional[str] = None
    checkout: Optional[str] = None
    guests: int = 2


async def search_booking(params: SearchParams) -> list[RentalListing]:
    """Search Booking.com via RapidAPI."""
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return []

    async with httpx.AsyncClient() as client:
        # First, get destination ID
        dest_response = await client.get(
            "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination",
            params={"query": params.location},
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "booking-com15.p.rapidapi.com"
            },
            timeout=30.0
        )

        if dest_response.status_code != 200:
            return []

        dest_data = dest_response.json()
        if not dest_data.get("data"):
            return []

        dest_id = dest_data["data"][0].get("dest_id")
        dest_type = dest_data["data"][0].get("dest_type", "city")

        # Search hotels
        search_params = {
            "dest_id": dest_id,
            "dest_type": dest_type,
            "adults": params.guests,
            "room_qty": 1,
            "page_number": 1,
            "units": "metric",
            "temperature_unit": "f",
            "languagecode": "en-us",
            "currency_code": "USD"
        }

        if params.checkin:
            search_params["arrival_date"] = params.checkin
        if params.checkout:
            search_params["departure_date"] = params.checkout

        hotels_response = await client.get(
            "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels",
            params=search_params,
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "booking-com15.p.rapidapi.com"
            },
            timeout=30.0
        )

        if hotels_response.status_code != 200:
            return []

        hotels_data = hotels_response.json()
        listings = []

        for hotel in hotels_data.get("data", {}).get("hotels", [])[:10]:
            prop = hotel.get("property", {})
            listings.append(RentalListing(
                source="booking.com",
                name=prop.get("name", "Unknown"),
                location=params.location,
                price=prop.get("priceBreakdown", {}).get("grossPrice", {}).get("value"),
                rating=prop.get("reviewScore"),
                url=f"https://www.booking.com/hotel/{prop.get('countryCode', '')}/{prop.get('ufi', '')}.html",
                image_url=prop.get("photoUrls", [None])[0] if prop.get("photoUrls") else None
            ))

        return listings


async def search_airbnb(params: SearchParams) -> list[RentalListing]:
    """Search Airbnb by scraping (similar to MCP server approach)."""
    async with httpx.AsyncClient() as client:
        url = f"https://www.airbnb.com/s/{params.location}/homes"
        query_params = {
            "adults": params.guests,
            "tab_id": "home_tab"
        }

        if params.checkin:
            query_params["checkin"] = params.checkin
        if params.checkout:
            query_params["checkout"] = params.checkout

        try:
            response = await client.get(
                url,
                params=query_params,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                follow_redirects=True,
                timeout=30.0
            )

            if response.status_code != 200:
                return []

            # Parse embedded JSON data from Airbnb page
            soup = BeautifulSoup(response.text, "html.parser")
            script_tags = soup.find_all("script", {"type": "application/json"})

            listings = []
            for script in script_tags:
                try:
                    import json
                    data = json.loads(script.string)
                    # Navigate Airbnb's nested data structure
                    if "niobeMinimalClientData" in str(data):
                        # Extract listings from the data structure
                        # This is simplified - actual implementation would need to
                        # traverse the nested structure
                        pass
                except:
                    continue

            # Fallback: return empty if parsing fails
            # In production, you'd use a more robust scraping approach
            return listings

        except Exception:
            return []


async def search_all(params: SearchParams) -> dict[str, list[RentalListing]]:
    """Search all sources in parallel."""
    import asyncio

    booking_task = asyncio.create_task(search_booking(params))
    airbnb_task = asyncio.create_task(search_airbnb(params))

    booking_results = await booking_task
    airbnb_results = await airbnb_task

    return {
        "booking": booking_results,
        "airbnb": airbnb_results
    }
