from typing import Any, Union
import asyncio
import httpx
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

server = Server("weather")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="get_alerts",
            description="Get weather alerts for a state",
            inputSchema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Two-letter state code (e.g. CA, NY)",
                    },
                },
                "required": ["state"],
            },
        ),
        types.Tool(
            name="get_forecast",
            description="Get weather forecast for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the location",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude of the location",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        ),
    ]

async def make_nws_request(client: httpx.AsyncClient, url: str) -> Union[dict[str, Any], None]:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }

    try:
        response = await client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a concise string."""
    props = feature["properties"]
    return (
        f"Event: {props.get('event', 'Unknown')}\n"
        f"Area: {props.get('areaDesc', 'Unknown')}\n"
        f"Severity: {props.get('severity', 'Unknown')}\n"
        f"Status: {props.get('status', 'Unknown')}\n"
        f"Headline: {props.get('headline', 'No headline')}\n"
        "---"
    )

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
    """
    Handle tool execution requests.
    Tools can fetch weather data and notify clients of changes.
    """
    if not arguments:
        raise ValueError("Missing arguments")
    if name == "get_alerts":
        state = arguments.get("state")
        if not state:
            raise ValueError("Missing state parameter")

        state = state.upper()
        if len(state) != 2:
            raise ValueError("State must be a two-letter code (e.g. CA, NY)")

        async with httpx.AsyncClient() as client:
            alerts_url = f"{NWS_API_BASE}/alerts?area={state}"
            alerts_data = await make_nws_request(client, alerts_url)

            if not alerts_data:
                return [types.TextContent(type="text", text="Failed to retrieve alerts data")]

            features = alerts_data.get("features", [])
            if not features:
                return [types.TextContent(type="text", text=f"No active alerts for {state}")]

            formatted_alerts = [format_alert(feature) for feature in features[:20]]
            alerts_text = f"Active alerts for {state}:\n\n" + "\n".join(formatted_alerts)

            return [
                types.TextContent(
                    type="text",
                    text=alerts_text
                )
            ]
    elif name == "get_forecast":
        try:
            latitude = float(arguments.get("latitude"))
            longitude = float(arguments.get("longitude"))
        except (TypeError, ValueError):
            return [types.TextContent(
                type="text",
                text="Invalid coordinates. Please provide valid numbers for latitude and longitude."
            )]

        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return [types.TextContent(
                type="text",
                text="Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180."
            )]

        async with httpx.AsyncClient() as client:
            points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
            points_data = await make_nws_request(client, points_url)

            if not points_data:
                return [types.TextContent(type="text", text=f"Failed to retrieve grid point data for coordinates: {latitude}, {longitude}. This location may not be supported by the NWS API (only US locations are supported).")]

            properties = points_data.get("properties", {})
            forecast_url = properties.get("forecast")

            if not forecast_url:
                return [types.TextContent(type="text", text="Failed to get forecast URL from grid point data")]

            forecast_data = await make_nws_request(client, forecast_url)

            if not forecast_data:
                return [types.TextContent(type="text", text="Failed to retrieve forecast data")]

            periods = forecast_data.get("properties", {}).get("periods", [])
            if not periods:
                return [types.TextContent(type="text", text="No forecast periods available")]

            formatted_forecast = []
            for period in periods:
                forecast_text = (
                    f"{period.get('name', 'Unknown')}:\n"
                    f"Temperature: {period.get('temperature', 'Unknown')}°{period.get('temperatureUnit', 'F')}\n"
                    f"Wind: {period.get('windSpeed', 'Unknown')} {period.get('windDirection', '')}\n"
                    f"{period.get('shortForecast', 'No forecast available')}\n"
                    "---"
                )
                formatted_forecast.append(forecast_text)

            forecast_text = f"Forecast for {latitude}, {longitude}:\n\n" + "\n".join(formatted_forecast)

            return [types.TextContent(
                type="text",
                text=forecast_text
            )]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="weather",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
