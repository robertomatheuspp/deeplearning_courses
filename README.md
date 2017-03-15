Este repositório contem alguns códigos apresentados durante mini-cursos sobre redes neurais convolucionais.

## Pré-requisitos: 
*Vontade de aprender  :-)*

----------
## Ferramentas Necessárias: 
 - Pyton (de preferência >= 3.*)
 - [Keras](https://keras.io):
	 - `sudo pip3 install keras` (verifique sua versão do pip).
	 - Para usuarios Windows este [link](http://stackoverflow.com/questions/34097988/how-do-i-install-keras-and-theano-in-anaconda-python-2-7-on-windows) e [este](http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/) podem ser uteis.
	 - Para usuários Windows, basta instalar o [Anaconda 32-Bit](https://www.continuum.io/downloads) e dentro do terminal do Anaconda utilizar os comandos pip install.
 - `pip3 install matplotlib pydot-ng python pillow h5py graphviz`. 
 - [Trocar backend keras do Tensorflow para o Theano](https://keras.io/backend/)
	 - `vi ./.keras/keras.json`
	 - `pressiona tecla 'i', onde tem "tensorflow" escrever "theano"`
	 - substitua também `"image_dim_ordering": "ft"` por `"image_dim_ordering": "th"`
	 - Para salvar pressione `esc` e então digite `wq` (write quit) e `pressione enter`.

Qualquer duvida, entre em [contato](mailto:robertomatheuspp@gmail.com).

----------
## Autores:
Projeto desenvolvido por:
[Roberto Pereira](http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K8743998Y2)
