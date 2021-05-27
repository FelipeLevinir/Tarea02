#ifndef __MATRIXTOMEM__
#define __MATRIXTOMEM__

/////////////////////////////////////////////////////////////////////////////////
//
//   Uso:
//   		MatrixToMem varName(fileName);
//
//   Description:
//                Carga los datos de una matriz 1d
//                desde el archivo 'fileName'
//                a memoria principal (variable 'varName')
//
//   Atributos públicos:
//	             _nfil : filas de datos de la matriz
//               _matrixInMemory: matriz en memoria.
//               operador [] sobrecargado para acceder a los datos de la matriz
//
/////////////////////////////////////////////////////////////////////////////////


class MatrixToMem{

private:
	std::ifstream _matrixFile;
	std::string   _fileName;
	
public:
	uint32_t _nfil;
	uint32_t *_matrixInMemory;
	
public:
	MatrixToMem(std::string fileName);
	~MatrixToMem();
	
	uint32_t operator[](uint32_t idx);
	
private:
	void openFile();
	void loadInMemory();

};

MatrixToMem::MatrixToMem(std::string fileName): _fileName(fileName)
{
	this->openFile();
	
	this->loadInMemory();
	
}


MatrixToMem::~MatrixToMem()
{
	delete[] _matrixInMemory;
}

void MatrixToMem::openFile()
{
	_matrixFile.open(_fileName);
	if(!_matrixFile.good()){
		std::cerr << "Error en acceso a archivo :" << _fileName <<std::endl;
		exit(EXIT_FAILURE);
	}
}

void MatrixToMem::loadInMemory()
{
	std::string line;
	size_t index = 0;
	
	std::getline(_matrixFile, line);
	
	_nfil    = std::stoi( line );
	_matrixInMemory    = new uint32_t[ _nfil ];
	
	for(size_t i = 0; i < _nfil; i++){
		std::getline(_matrixFile, line);

		_matrixInMemory[i] = std::stoi( line );
	}
	_matrixFile.close();
}

uint32_t MatrixToMem::operator[](uint32_t idx)
{
	return(_matrixInMemory[idx]);
}


#endif