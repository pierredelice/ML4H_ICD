from flask import Flask, request, jsonify
from flask import Flask
#from flask_swagger_ui import get_swaggerui_blueprint
from flask_restx import Api, Resource, reqparse
#import joblib
# 


app = Flask(__name__)
api = Api(app, version='1.0',
           title='ICD API',
           description='API for ICD')

ns = api.namespace('icd', description='Clasificación ICD')

@ns.route('/hello')
class HelloWorld(Resource):
    def get(self):

        return {'hello': 'API para la codificación de un texto libre en clave ICD'}


if __name__ == '__main__':
    app.run(debug=True)