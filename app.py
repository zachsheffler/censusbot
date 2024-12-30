from flask import Flask, request, Response, render_template
from pums_readers import graph
from langchain_core.messages import HumanMessage

# Make sure to import or initialize `graph` here if needed
# from your_module import graph

app = Flask(__name__)

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    """
    This route renders a simple form where users can
    input text that is then submitted via POST request.
    """
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    """
    This route receives the text from the form and
    renders a page displaying the submitted text.
    """
    user_input = request.form.get('user_input', '')

    final_state = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]}
    )

    return render_template('result.html', user_text=user_input, response=final_state["messages"][-1].content)

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == "__main__":
    # Turn off debug in production
    app.run(debug=True)
