from flask import Flask, render_template, request
from EventCollect import retrieve_Sents, common_tokens, analyze, parse
import operator
import time
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/list', methods=['GET', 'POST'])
def list():
	start_t = time.time_ns()
	if request.method == 'POST':
		inp = request.form["search"]
	sents = parse(inp)
	common = common_tokens(inp)
	finalSents = retrieve_Sents(sents, True)
	result = analyze(common, finalSents)

	result.sort(key = operator.itemgetter(1), reverse=True)

	result_sig = result

	result_sig.sort(key = operator.itemgetter(2))

	result_date = result_sig

	end_t = time.time_ns()

	offset = (end_t - start_t)

	print("Time took "+ str(offset) + " seconds")
	
	return render_template('time.html', result_date=result_date, inp=inp, common=common[0:10])

if __name__ == '__main__':
   app.run()