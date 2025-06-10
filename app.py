import gradio as gr
import time

from union.remote import UnionRemote


remote = UnionRemote()
workflow = remote.fetch_workflow(name="llmops_rag.rag_basic.rag_basic")


N_RETRIES = 200


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    if message is not None:
        history.append({"role": "user", "content": message})
    return history, gr.Textbox(value=None, interactive=False)


def bot(history: list):
    last_user_message = [msg for msg in history if msg["role"] == "user"][-1]["content"]
    execution = remote.execute(workflow, inputs={"questions": [last_user_message]})
    url = remote.generate_console_url(execution)
    print(f"🚀 Union Serverless execution url: {url}")

    answers = None
    for _ in range(N_RETRIES):
        # gets the answer from the first node, which is the "ask" workflow.
        # the second part of the workflow is the feedback loop.
        if execution.is_done:
            answers = execution.outputs["o0"]
            break
        execution = remote.sync(execution, sync_nodes=True)
        time.sleep(1)

    if answers is None:
        raise RuntimeError("Failed to get answer")
    
    answer = answers[0]

    history.append({"role": "assistant", "content": ""})
    history[-1]["content"] += answer
    yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chatbot", height="80vh", bubble_full_width=False, type="messages")

    chat_input = gr.Textbox(
        interactive=True,
        placeholder="Enter message...",
        show_label=False,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.Textbox(interactive=True), None, [chat_input])


if __name__ == "__main__":
    demo.launch(debug=True, share=True)
