from flask import Flask, request, jsonify
from flask import Flask
#from flask_swagger_ui import get_swaggerui_blueprint
from flask_restx import Api, Resource, reqparse
#import joblib
# 
from modelos import BiGru
modelo = BiGru('.','.')

app = Flask(__name__)
api = Api(app, version='1.0',
           title='ICD API',
           description='API for ICD')

ns = api.namespace('icd', description='Clasificación ICD')

parser = reqparse.RequestParser()
parser.add_argument('texto', type = str, help = 'Texto a clasificar')

@ns.route('/modelo/bigru/predict')
class BiGru(Resource):
    @api.doc(parser = parser)
    def get(self):
        text = parser.parse_args()
        prediccion = modelo.predict_batch(text['texto'])
        return jsonify({'causa': prediccion, 
                        'texto' : text['texto']})
    
@ns.route('/hello')
class HelloWorld(Resource):
    def get(self):

        return {'hello': 'Hola Mundo de api'}


if __name__ == '__main__':
    app.run(debug=True)