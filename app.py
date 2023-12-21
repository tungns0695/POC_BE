from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from itertools import groupby
from operator import itemgetter
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        data = request.get_json()

        # Extract 'k' and 'orders' from the request
        k = data.get('k')
        orders = data.get('orders')

        # Extract latitude and longitude from orders for clustering
        coordinates = [(order['lat'], order['long']) for order in orders]

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(coordinates)

        # Add 'group' field to each order in the response
        for i, order in enumerate(orders):
            order['group'] = f'group{labels[i]+1}'  # Groups start from 1

        orders.sort(key=itemgetter('group'))
        grouped_orders = {group: list(orders) for group, orders in groupby(orders, key=itemgetter('group'))}

        # Return the modified request
        response_data = {
            'k': k,
            'orders': grouped_orders
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()