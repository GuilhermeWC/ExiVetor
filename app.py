from flask import Flask, request, redirect, url_for, render_template, flash
import mxnet as mx
import numpy as np
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
import os
import array
import oracledb
from werkzeug.utils import secure_filename
from collections import namedtuple

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Chave secreta para sessões e mensagens de feedback

# Diretórios para uploads
UPLOAD_FOLDER = 'static/uploads'
SEARCH_FOLDER = 'static/search'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEARCH_FOLDER'] = SEARCH_FOLDER

def allowed_file(filename):
    """ Verifica se a extensão do arquivo é permitida """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image(path):
    """ Carrega a imagem do caminho fornecido """
    img = mx.image.imread(path)
    if img is None:
        return None
    return img

def preprocess(img):
    """ Pré-processa a imagem para a inferência no modelo """
    transform_fn = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(axis=0)
    return img

def gen_embeddings(path):
    """ Gera embeddings para a imagem fornecida """
    img = get_image(path)
    img = preprocess(img)
    mod.forward(Batch([img]))
    scores = mx.ndarray.softmax(mod.get_outputs()[0]).asnumpy()
    return array.array("f", scores.squeeze())

def search_similar_images(vector_data_32):
    """ Busca imagens semelhantes no banco de dados """
    connection = oracledb.connect(
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        dsn=os.getenv('DB_DSN')
    )
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, filename, description FROM image_vector "
            "ORDER BY vector_distance(v32, :1, COSINE) "
            "FETCH FIRST 3 ROWS ONLY",
            [vector_data_32]
        )
        results = cursor.fetchall()
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        description = request.form.get('description')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Gerar embedding e inserir no banco de dados
            vector_data_32 = gen_embeddings(file_path)

            try:
                connection = oracledb.connect(
                    user=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD'),
                    dsn=os.getenv('DB_DSN')
                )
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO image_vector (id, filename, description, v32) "
                        "VALUES (image_vector_seq.NEXTVAL, :1, :2, :3)",
                        (filename, description, vector_data_32)
                    )
                    connection.commit()
                flash('Imagem carregada com sucesso!', 'success')
            except Exception as e:
                flash(f'Erro ao carregar imagem: {e}', 'error')

            return redirect(url_for('index'))
        else:
            flash('Arquivo não permitido ou falta de descrição.', 'error')

    return render_template('upload.html')

@app.route('/search', methods=['POST'])
def search():
    file = request.files.get('search_file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['SEARCH_FOLDER'], filename)
        file.save(file_path)

        # Gerar embedding e buscar imagens semelhantes
        vector_data_32 = gen_embeddings(file_path)
        results = search_similar_images(vector_data_32)

        return render_template('upload.html', results=results)
    else:
        flash('Arquivo não permitido para busca.', 'error')
        return redirect(url_for('index'))

# Inicialização do modelo ONNX
sym, arg_params, aux_params = import_model('resnet18-v2-7.onnx')

# Determinar o contexto (CPU ou GPU)
if len(mx.test_utils.list_gpus()) == 0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)

# Configurar o módulo MXNet para inferência
mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

# Nome da namedtuple para os dados do batch
Batch = namedtuple('Batch', ['data'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Configurar para aceitar conexões externas
