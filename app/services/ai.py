import os
import json
from anthropic import Anthropic
from .rentals import search_all, search_booking, SearchParams


client = None


def get_client() -> Anthropic:
    global client
    if client is None:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client


SYSTEM_PROMPT = """You are a helpful AI trip planner assistant. You help users find vacation rentals and plan their trips.

You have access to search tools for finding accommodations:
- Booking.com: Hotels and vacation rentals worldwide
- Airbnb: Vacation rentals and unique stays

When users ask about finding places to stay, extract the following information:
- Location/destination
- Check-in date (YYYY-MM-DD format)
- Check-out date (YYYY-MM-DD format)
- Number of guests

Then use the search_rentals function to find options for them.

Be conversational and helpful. Ask clarifying questions if needed. After showing results, offer to help refine the search or provide more details about specific listings.

Format prices clearly and highlight key features like ratings, location, and amenities when available."""


TOOLS = [
    {
        "name": "search_rentals",
        "description": "Search for vacation rentals and hotels across multiple platforms (Booking.com, Airbnb). Returns listings with prices, ratings, and availability.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The destination city or area (e.g., 'Miami Beach', 'Paris', 'Tokyo')"
                },
                "checkin": {
                    "type": "string",
                    "description": "Check-in date in YYYY-MM-DD format"
                },
                "checkout": {
                    "type": "string",
                    "description": "Check-out date in YYYY-MM-DD format"
                },
                "guests": {
                    "type": "integer",
                    "description": "Number of guests",
                    "default": 2
                }
            },
            "required": ["location"]
        }
    }
]


async def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result."""
    if tool_name == "search_rentals":
        params = SearchParams(
            location=tool_input.get("location", ""),
            checkin=tool_input.get("checkin"),
            checkout=tool_input.get("checkout"),
            guests=tool_input.get("guests", 2)
        )
        results = await search_all(params)

        # Format results for the AI
        formatted = []
        for source, listings in results.items():
            if listings:
                formatted.append(f"\n## {source.title()} Results:")
                for i, listing in enumerate(listings[:5], 1):
                    formatted.append(f"\n{i}. **{listing.name}**")
                    if listing.price:
                        formatted.append(f"   - Price: ${listing.price}")
                    if listing.rating:
                        formatted.append(f"   - Rating: {listing.rating}/10")
                    if listing.location:
                        formatted.append(f"   - Location: {listing.location}")
                    if listing.url:
                        formatted.append(f"   - [View listing]({listing.url})")

        if not formatted:
            return "No results found. Try a different location or dates."

        return "\n".join(formatted)

    return "Unknown tool"


async def chat(messages: list[dict], user_message: str) -> tuple[str, list[dict]]:
    """Process a chat message and return the response."""
    anthropic = get_client()

    # Add user message to history
    messages.append({"role": "user", "content": user_message})

    # Initial API call
    response = anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages
    )

    # Handle tool use loop
    while response.stop_reason == "tool_use":
        # Extract tool use from response
        tool_use_block = next(
            (block for block in response.content if block.type == "tool_use"),
            None
        )

        if not tool_use_block:
            break

        # Execute the tool
        tool_result = await process_tool_call(
            tool_use_block.name,
            tool_use_block.input
        )

        # Add assistant's response with tool use to messages
        messages.append({"role": "assistant", "content": response.content})

        # Add tool result
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result
            }]
        })

        # Continue the conversation
        response = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

    # Extract final text response
    text_content = next(
        (block.text for block in response.content if hasattr(block, "text")),
        "I couldn't generate a response."
    )

    # Add final assistant message
    messages.append({"role": "assistant", "content": text_content})

    return text_content, messages
