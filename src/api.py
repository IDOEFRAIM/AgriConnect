from flask import Flask, jsonify, request
from qa import qaReply, qaState
from typing import TypedDict , List

app = Flask(__name__)

class PerformanceState(TypedDict):
    question:str
    anwser:str
    time:str

performance: List[PerformanceState] = []

@app.route('/reply', methods=['POST'])
def reply():
    print('starting')
    data = request.get_json()
    query = data.get("query", "")
    chatHistory = data.get("chatHistory", [])

    if not isinstance(chatHistory, list):
        chatHistory = []

    questionState = qaState(query)

    state,timeToReply = qaReply(questionState)
    # I add performance for feature evaluations
    performance.append(
        {
            "question":state.get("question",""),
            "answer":state["answer"],
            "time":timeToReply
        }
    )

    chatHistory.append({"role": "user", "content": query})
    chatHistory.append({"role": "assistant", "content": state["answer"]})

    return jsonify({ "chatHistory": chatHistory})


# We can use this endpoint to get the performance
@app.route('/performance',methods=['GET'])
def getPerf():
    return jsonify(performance)


if __name__ == "__main__":
    app.url_map.strict_slashes = False
    app.run(host="localhost", port=5000, debug=True)