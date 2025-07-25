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
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": query
            }
        ],
        "tools": [
            {
                "type": "web_search"
            }
        ]
    }
    response = requests.post(url, headers=headers, json=payload, verify=False)
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


# sample_json = {

#     "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDpiMGI1OTgzZjI1MTViNzk4ZTIxNTM2MGFjYWRlMWMyZmE1OGJhNzQ3NjFmYTc4ZDI0YzI5ZGY3MDdhM2RlYWZlO3Jlc3BvbnNlX2lkOnJlc3BfNjg4MzkyMTAxNDMwODE5MjlmZmZlMjI5MDQwMTNmZWYwYTI2MzZlOWM5MzVhYjZh",
#     "created_at": 1753453072.0,
#     "error": "None",
#     "incomplete_details": "None",
#     "instructions": "None",
#     "metadata": {

#     },
#     "model": "gpt-4o-2024-08-06",
#     "object": "response",
#     "output": [
#         {
#             "id": "ws_68839210ce4c8192aaa7e0c4e7caf4560a2636e9c935ab6a",
#             "action": {
#                 "query": "latest AI developments 2023",
#                 "type": "search",
#                 "domains": "None"
#             },
#             "status": "completed",
#             "type": "web_search_call"
#         },
#         {
#             "id": "msg_6883921430388192bf1e8e5fa06ec9ae0a2636e9c935ab6a",
#             "content": [
#                 {
#                     "annotations": [
#                         {
#                             "end_index": 640,
#                             "start_index": 547,
#                             "title": "Google's AI fight is moving to new ground",
#                             "type": "url_citation",
#                             "url": "https://www.ft.com/content/2bb07757-6039-46cc-8f1a-73d0a83584a7?utm_source=openai"
#                         },
#                         {
#                             "end_index": 1082,
#                             "start_index": 957,
#                             "title": "AI lottery-ticket scramble sparks an R&D spiral",
#                             "type": "url_citation",
#                             "url": "https://www.reuters.com/default/ai-lottery-ticket-scramble-sparks-an-rd-spiral-2025-07-24/?utm_source=openai"
#                         },
#                         {
#                             "end_index": 1502,
#                             "start_index": 1383,
#                             "title": "AI Intelligencer How AI won math gold",
#                             "type": "url_citation",
#                             "url": "https://www.reuters.com/technology/ai-intelligencer-how-ai-won-math-gold-2025-07-24/?utm_source=openai"
#                         },
#                         {
#                             "end_index": 2000,
#                             "start_index": 1835,
#                             "title": "Atlantic allies turn to xAI and OpenAI - Why chatbots are now a matter of national strategy",
#                             "type": "url_citation",
#                             "url": "https://www.windowscentral.com/artificial-intelligence/openai-signs-uk-government-deal-while-xai-lands-u-s-defense-contract?utm_source=openai"
#                         },
#                         {
#                             "end_index": 2274,
#                             "start_index": 2181,
#                             "title": "A good day for AI, a rough one for Musk",
#                             "type": "url_citation",
#                             "url": "https://www.ft.com/content/37d4b9ca-7034-4ed7-82cb-4fd1a52b434a?utm_source=openai"
#                         },
#                         {
#                             "end_index": 2735,
#                             "start_index": 2551,
#                             "title": "Windows 11 gets new AI-powered features in latest update - here's 4 tools to try out now",
#                             "type": "url_citation",
#                             "url": "https://www.tomsguide.com/computing/windows-operating-systems/windows-11-gets-new-ai-powered-features-in-latest-update-heres-4-tools-to-try-out-now?utm_source=openai"
#                         },
#                         {
#                             "end_index": 3085,
#                             "start_index": 2983,
#                             "title": "Latest AI Trends: Key Developments Shaping the Future",
#                             "type": "url_citation",
#                             "url": "https://rytsensetech.com/ai-development/latest-ai-development/?utm_source=openai"
#                         },
#                         {
#                             "end_index": 3424,
#                             "start_index": 3322,
#                             "title": "Latest AI Trends: Key Developments Shaping the Future",
#                             "type": "url_citation",
#                             "url": "https://rytsensetech.com/ai-development/latest-ai-development/?utm_source=openai"
#                         },
#                         {
#                             "end_index": 3755,
#                             "start_index": 3653,
#                             "title": "Latest AI Trends: Key Developments Shaping the Future",
#                             "type": "url_citation",
#                             "url": "https://rytsensetech.com/ai-development/latest-ai-development/?utm_source=openai"
#                         },
#                         {
#                             "end_index": 4118,
#                             "start_index": 4014,
#                             "title": "Artificial intelligence visual art",
#                             "type": "url_citation",
#                             "url": "https://en.wikipedia.org/wiki/Artificial_intelligence_visual_art?utm_source=openai"
#                         },
#                         {
#                             "end_index": 4460,
#                             "start_index": 4367,
#                             "title": "Artificial intelligence",
#                             "type": "url_citation",
#                             "url": "https://en.wikipedia.org/wiki/Artificial_intelligence?utm_source=openai"
#                         },
#                         {
#                             "end_index": 4888,
#                             "start_index": 4778,
#                             "title": "AI in 2025: Top Trends and Developments",
#                             "type": "url_citation",
#                             "url": "https://www.mirantis.com/blog/the-state-of-ai-key-developments-and-trends/?utm_source=openai"
#                         },
#                         {
#                             "end_index": 5049,
#                             "start_index": 4923,
#                             "title": "Google's AI fight is moving to new ground",
#                             "type": "url_citation",
#                             "url": "https://www.ft.com/content/2bb07757-6039-46cc-8f1a-73d0a83584a7?utm_source=openai"
#                         },
#                         {
#                             "end_index": 5211,
#                             "start_index": 5052,
#                             "title": "AI lottery-ticket scramble sparks an R&D spiral",
#                             "type": "url_citation",
#                             "url": "https://www.reuters.com/default/ai-lottery-ticket-scramble-sparks-an-rd-spiral-2025-07-24/?utm_source=openai"
#                         },
#                         {
#                             "end_index": 5357,
#                             "start_index": 5214,
#                             "title": "AI Intelligencer How AI won math gold",
#                             "type": "url_citation",
#                             "url": "https://www.reuters.com/technology/ai-intelligencer-how-ai-won-math-gold-2025-07-24/?utm_source=openai"
#                         }
#                     ],
#                     "text": "Artificial intelligence (AI) continues to evolve rapidly, impacting various sectors and prompting significant investments and policy developments. Here are some of the latest advancements as of July 25, 2025:"
#                     "\n\n**1. Corporate Investments and Competition**"
#                     "\n\n- **Google's AI Integration**: Google has incorporated AI-generated answers into its search results, leading to a 10% increase in relevant queries without affecting advertising revenue. The company is also focusing on developing AI applications like Gemini to compete with OpenAI's ChatGPT. ([ft.com](https://www.ft.com/content/2bb07757-6039-46cc-8f1a-73d0a83584a7?utm_source=openai))"
#                     "\n\n- **R&D Expenditure Surge**: Tech giants such as Meta plan to invest heavily in AI research and development, with Meta allocating $60 billion for capital expenditures and over $50 billion for R&D in 2025. This intense competition has led to substantial bonuses for AI talent, sometimes reaching up to $100 million. ([reuters.com](https://www.reuters.com/default/ai-lottery-ticket-scramble-sparks-an-rd-spiral-2025-07-24/?utm_source=openai))"
#                     "\n\n**2. AI Achievements in Mathematics**\n\n- **International Mathematical Olympiad**: In July 2025, AI models from Google DeepMind and OpenAI achieved gold-medal status at the International Mathematical Olympiad, demonstrating significant progress in machine reasoning and problem-solving capabilities. ([reuters.com](https://www.reuters.com/technology/ai-intelligencer-how-ai-won-math-gold-2025-07-24/?utm_source=openai))"
#                     "\n\n**3. Government Collaborations and Policies**"
#                     "\n\n- **UK and US AI Initiatives**: The UK government has partnered with OpenAI to integrate advanced AI into public services, while the U.S. Department of Defense awarded contracts to AI companies, including OpenAI and Elon Musk's xAI, to develop AI workflows for military applications. ([windowscentral.com](https://www.windowscentral.com/artificial-intelligence/openai-signs-uk-government-deal-while-xai-lands-u-s-defense-contract?utm_source=openai)"
#                     "\n\n- **U.S. AI Action Plan**: The U.S. administration unveiled a 28-page AI action plan emphasizing deregulation and increased tech exports to enhance competitiveness against China. ([ft.com](https://www.ft.com/content/37d4b9ca-7034-4ed7-82cb-4fd1a52b434a?utm_source=openai))"
#                     "\n\n**4. AI Integration in Consumer Technology**"
#                     "\n\n- **Windows 11 AI Features**: Microsoft's latest Windows 11 update introduces AI-powered tools such as the AI Settings Agent for natural language system adjustments and the Photos App Relight Feature for enhanced photo lighting. ([tomsguide.com](https://www.tomsguide.com/computing/windows-operating-systems/windows-11-gets-new-ai-powered-features-in-latest-update-heres-4-tools-to-try-out-now?utm_source=openai))"
#                     "\n\n**5. Advances in AI Applications**"
#                     "\n\n- **On-Device AI**: There's a shift towards deploying AI models directly on devices like smartphones and wearables, enhancing speed, privacy, and cost-efficiency by reducing reliance on cloud-based processing. ([rytsensetech.com](https://rytsensetech.com/ai-development/latest-ai-development/?utm_source=openai))"
#                     "\n\n- **Responsible AI Development**: Companies are focusing on responsible and explainable AI, implementing measures to disclose decision-making processes, detect biases, and ensure human oversight, aligning with regulatory expectations. ([rytsensetech.com](https://rytsensetech.com/ai-development/latest-ai-development/?utm_source=openai))"
#                     "\n\n- **AI in Cybersecurity**: AI-driven cybersecurity systems are being developed to detect behavioral anomalies, predict attack vectors, and automate incident responses, addressing the increasing sophistication of cyber threats. ([rytsensetech.com](https://rytsensetech.com/ai-development/latest-ai-development/?utm_source=openai))"
#                     "\n\n**6. AI in Creative Industries**"
#                     "\n\n- **Advancements in AI-Generated Art**: Models like OpenAI's GPT Image 1 and Google's Imagen 4 have been released, offering improved text rendering and photorealism in AI-generated images, expanding creative possibilities. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Artificial_intelligence_visual_art?utm_source=openai))"
#                     "\n\n**7. AI in Healthcare**"
#                     "\n\n- **Drug Discovery**: AI has accelerated the search for Parkinson's disease treatments by identifying compounds that prevent protein aggregation, speeding up the screening process tenfold and reducing costs significantly. ([en.wikipedia.org](https://en.wikipedia.org/wiki/Artificial_intelligence?utm_source=openai))"
#                     "\n\n**8. Robotics and Embodied AI**"
#                     "\n\n- **Advancements in Robotics**: The integration of AI with physical systems has led to significant progress in robotics, with platforms like NEO Gamma and Tesla's Optimus Gen 3 demonstrating enhanced dexterity and autonomy, impacting industries such as manufacturing and healthcare. ([mirantis.com](https://www.mirantis.com/blog/the-state-of-ai-key-developments-and-trends/?utm_source=openai))"
#                     "\n\n\n## Recent Developments in AI:\n- [Google's AI fight is moving to new ground](https://www.ft.com/content/2bb07757-6039-46cc-8f1a-73d0a83584a7?utm_source=openai)"
#                     "\n- [AI lottery-ticket scramble sparks an R&D spiral](https://www.reuters.com/default/ai-lottery-ticket-scramble-sparks-an-rd-spiral-2025-07-24/?utm_source=openai)"
#                     "\n- [AI Intelligencer How AI won math gold](https://www.reuters.com/technology/ai-intelligencer-how-ai-won-math-gold-2025-07-24/?utm_source=openai) ",
#                     "type": "output_text",
#                     "logprobs": [

#                     ]
#                 }
#             ],
#             "role": "assistant",
#             "status": "completed",
#             "type": "message"
#         }
#     ],
#     "parallel_tool_calls": True,
#     "temperature": 1.0,
#     "tool_choice": "auto",
#     "tools": [
#         {
#             "type": "web_search_preview",
#             "search_context_size": "medium",
#             "user_location": {
#                     "type": "approximate",
#                     "city": "None",
#                     "country": "US",
#                     "region": "None",
#                     "timezone": "None"
#             }
#         }
#     ],
#     "top_p": 1.0,
#     "max_output_tokens": "None",
#     "previous_response_id": "None",
#     "reasoning": {
#         "effort": "None",
#         "summary": "None"
#     },
#     "status": "completed",
#     "text": {
#         "format": {
#             "type": "text"
#         }
#     },
#     "truncation": "disabled",
#     "usage": {
#         "input_tokens": 304,
#         "input_tokens_details": {
#             "audio_tokens": "None",
#             "cached_tokens": 0,
#             "text_tokens": "None"
#         },
#         "output_tokens": 1239,
#         "output_tokens_details": {
#             "reasoning_tokens": 0,
#             "text_tokens": "None"
#         },
#         "total_tokens": 1543
#     },
#     "user": "None"
# }


# results = extract_links_titles_snippets(sample_json)
# print(results, 'results')
