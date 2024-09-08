import os
from flask import Flask, render_template, request, send_from_directory
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from integraltest import IntegralQueryParser, compute_integral, compute_and_visualize_integral, parse_function, determine_best_integral

# Get the absolute path of the directory containing this file
basedir = os.path.abspath(os.path.dirname(__file__))

# Create the Flask app and set the template folder
app = Flask(__name__, template_folder=os.path.join(basedir, 'templates'))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    plot_url = None
    query = ''
    function = ''
    lower_bound = upper_bound = None
    method = ''
    error_message = None

    if request.method == 'POST':
        query = request.form.get('query', '')
        
        if query:
            try:
                query_parser = IntegralQueryParser()
                intent = query_parser.parse_query(query)
                
                if intent['function'] is None:
                    error_message = "I'm sorry, I couldn't understand the function you want to integrate."
                elif intent['lower_bound'] is None or intent['upper_bound'] is None:
                    error_message = "I'm sorry, I couldn't determine the bounds of integration."
                else:
                    function = intent['function']
                    lower_bound = intent['lower_bound']
                    upper_bound = intent['upper_bound']
                    
                    visualize = 'visualize' in query.lower()
                    if visualize:
                        result, method = compute_and_visualize_integral(function, lower_bound, upper_bound)
                    else:
                        result, method = compute_integral(function, lower_bound, upper_bound)
                    
                    # Generate plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.linspace(lower_bound, upper_bound, 1000)
                    func = parse_function(function)
                    y = [func(xi) for xi in x]
                    ax.plot(x, y, label='f(x)')
                    ax.fill_between(x, y, alpha=0.3)
                    
                    if method == 'riemann':
                        n = 20  # number of rectangles for visualization
                        dx = (upper_bound - lower_bound) / n
                        x_riemann = np.linspace(lower_bound, upper_bound, n+1)
                        y_riemann = [func(xi) for xi in x_riemann]
                        ax.bar(x_riemann[:-1], y_riemann[:-1], width=dx, alpha=0.5, align='edge', color='r', label='Riemann sum')
                    elif method == 'lebesgue':
                        y_min, y_max = min(y), max(y)
                        levels = np.linspace(y_min, y_max, 10)
                        for level in levels:
                            ax.axhline(y=level, color='r', alpha=0.2)
                    
                    ax.set_title(f'Integral of {function} from {lower_bound} to {upper_bound}')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.legend()
                    
                    # Save plot to a buffer
                    buf = io.BytesIO()
                    FigureCanvas(fig).print_png(buf)
                    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
                    plt.close(fig)
                    
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
        else:
            error_message = "Please enter a query."

    return render_template('index.html', result=result, plot_url=plot_url, 
                           query=query, function=function, 
                           lower_bound=lower_bound, upper_bound=upper_bound, 
                           method=method, error_message=error_message)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

