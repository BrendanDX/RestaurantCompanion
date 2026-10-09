"""Lab 10: FastMCP server exposing the culinary map resource and three restaurant tools.

Run standalone:  python -m connoisseur.mcp_app.server
(Normally it is launched automatically over stdio by client.py / chat_app.py.)

NOTE: stdio servers must never print to stdout - that channel carries the protocol.
"""
from fastmcp import FastMCP

from connoisseur.mcp_app import tools

mcp = FastMCP("Connoisseur-Server")

mcp.resource("culinary-map://california")(tools.get_culinary_map)
mcp.tool()(tools.get_restaurant_info)
mcp.tool()(tools.recommend_by_vibe)
mcp.tool()(tools.get_review)

if __name__ == "__main__":
    mcp.run()
