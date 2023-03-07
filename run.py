from flask import Flask, render_template, request
from EventCollect import retrieve_Sents, common_tokens, analyze, parse
import operator
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/list', methods=['GET', 'POST'])
def list():
	if request.method == 'POST':
		inp = request.form["search"]
	sents = parse("History of " + inp)
	common = common_tokens("History of " + inp)
	finalSents = retrieve_Sents(sents)
	result = analyze(common, finalSents)

	result.sort(key = operator.itemgetter(1), reverse=True)

	result_sig = result

	result_sig.sort(key = operator.itemgetter(2))

	result_date = result_sig
	
	return render_template('time.html', result_date=result_date, inp=inp, common=common[0:10])

if __name__ == '__main__':
   app.run()