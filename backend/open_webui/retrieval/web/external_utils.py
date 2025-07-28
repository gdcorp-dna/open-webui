import requests

from open_webui.retrieval.web.main import SearchResult

from fastapi import (
    Request
)


def call_external_api(query: str, request: Request, model: str = "gpt-4o") -> list[SearchResult]:
    """
    Calls the external API with the given query, token, url, and model name.
    Returns the JSON response.
    """
    token = request.app.state.config.EXTERNAL_WEB_SEARCH_API_KEY
    url = request.app.state.config.EXTERNAL_WEB_SEARCH_URL
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # old pattern
    # payload = {
    #     "model": model,
    #     "input": [
    #         {
    #             "role": "user",
    #             "content": query
    #         }
    #     ],
    #     "tools": [
    #         {
    #             "type": "web_search"
    #         }
    #     ]
    # }
    # # new
    payload = {
        "model": model,
        "input": query,
        "tools": [
            {
                "type": "web_search"
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload, verify=True)
    response.raise_for_status()
    return search_external_api_custom(response.json())


def search_external_api_custom(data):
    """
    Extracts an array of dicts with keys: link, title, snippet (empty string)
    from the output -> content -> annotations property of the provided JSON.
    """
    results = []
    output = data.get("output", [])
    for item in output:
        # Only process items with 'content' key
        content_list = item.get("content", [])
        for content in content_list:
            for annotation in content.get("annotations", []):
                results.append({
                    "link": annotation.get("url", ""),
                    "title": annotation.get("title", ""),
                    "snippet": ""
                })
    return results