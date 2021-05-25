#include <global.hh>

#include <RandomUnifStream.hpp>
#include <Timing.hpp>
#include <MatrixToMem.hpp>

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

/////////////////////////////////////////////////////////////////////////////////
//   Usage:
//           ./program_name  .......
//
//   Description:
//                ...................
//
/////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////
//  Sorting Network
//      **********************************************************
//      ***Importante: el paso de parámetros es por referencia.***
//      **********************************************************
//
//  Input:
//      __m128i*  dataRegisters  : Arreglo de 4 vectores __m128i, cada uno
//                                con una secuencia desordenada de 4 enteros
//
//  Output:
//    La secuencia de 4 enteros  ordenada de cada vector se almacena
//    en las columnas del arreglo 'dataRegisters'.
//
void sortNet(__m128i* dataRegisters) {

	__m128i r_min1, r_max1, r_min2, r_max2, r_min3, r_max3, r_min4, r_max4, r_min5, r_max5;
	//Paso 1
	r_min1 = _mm_min_epi32(dataRegisters[0], dataRegisters[2]);
	r_max1 = _mm_max_epi32(dataRegisters[0], dataRegisters[2]);
	//Paso 2
	r_min2 = _mm_min_epi32(dataRegisters[1], dataRegisters[3]);
	r_max2 = _mm_max_epi32(dataRegisters[1], dataRegisters[3]);
	//Paso 3
	r_min3 = _mm_min_epi32(r_max1, r_max2);
	r_max3 = _mm_max_epi32(r_max1, r_max2);
	//Paso 4
	r_min4 = _mm_min_epi32(r_min1, r_min2);
	r_max4 = _mm_max_epi32(r_min1, r_min2);
	//Paso 5
	r_min5 = _mm_min_epi32(r_min3, r_max4);
	r_max5 = _mm_max_epi32(r_min3, r_max4);

	dataRegisters[0] = r_min4;
	dataRegisters[1] = r_min5;
	dataRegisters[2] = r_max5;
	dataRegisters[3] = r_max3;
}

//////////////////////////////////////////////////////////////////////
// transpose a matrix vector
//      **********************************************************
//      ***Importante: el paso de parámetros es por referencia.***
//      **********************************************************
//   Input:
//       __m128i*  dataReg  : Arreglo de 4 vectores __m128i
//   Output:
//       __m128i*  dataReg  : Arreglo de 4 vectores que es 
//                            la matriz transpuesta de la original
//
void transpose(__m128i*  dataReg){
	__m128i S[4];

	S[0] = _mm_unpacklo_epi32(dataReg[0], dataReg[1]);
	S[1] = _mm_unpacklo_epi32(dataReg[2], dataReg[3]);
	S[2] = _mm_unpackhi_epi32(dataReg[0], dataReg[1]);
	S[3] = _mm_unpackhi_epi32(dataReg[2], dataReg[3]);

	dataReg[0] = _mm_unpacklo_epi64(S[0], S[1]);
	dataReg[1] = _mm_unpackhi_epi64(S[0], S[1]);
	dataReg[2] = _mm_unpacklo_epi64(S[2], S[3]);
	dataReg[3] = _mm_unpackhi_epi64(S[2], S[3]);
	
}

//////////////////////////////////////////////////////////////////////
//  Bitonic sorter
//      **********************************************************
//      ***Importante: el paso de parámetros es por referencia.***
//      **********************************************************
//  Input:
//      __m128i*  dataReg1  : secuencia ordenada de 4 enteros ascedente
//      __m128i*  dataReg2  : secuencia ordenada de 4 enteros ascedente
//
//  Output:
//    La secuencia de 8 enteros totalmente ordenada se almacena en:
//      __m128i*  dataReg1   
//      __m128i*  dataReg2 
//
void bitonicSorter(__m128i*  dataReg1, __m128i*  dataReg2)
{
	*dataReg2=_mm_shuffle_epi32(*dataReg2, _MM_SHUFFLE(0, 1, 2, 3));
	auto aux=_mm_min_epi32(*dataReg1,*dataReg2);
	*dataReg2=_mm_max_epi32(*dataReg1,*dataReg2);
	*dataReg1=aux;
	//Reordenar dataReg2 para que la entrada sea una secuencia bitónica
	//dataReg2 = _mm_shuffle_epi32(*dataReg2, _MM_SHUFFLE(0, 1, 2, 3));
	//std::cout << "pasa 1" << std::endl;
	uint32_t m1=_mm_extract_epi32(*dataReg1,0);
    uint32_t m2=_mm_extract_epi32(*dataReg1,1);
    uint32_t m3=_mm_extract_epi32(*dataReg1,2);
    uint32_t m4=_mm_extract_epi32(*dataReg1,3);
/////////////////////////////////////////////////////////////////////////
    uint32_t M1=_mm_extract_epi32(*dataReg2,0);
    uint32_t M2=_mm_extract_epi32(*dataReg2,1);
    uint32_t M3=_mm_extract_epi32(*dataReg2,2);
    uint32_t M4=_mm_extract_epi32(*dataReg2,3);
////////////////////////////////////////////////////////////////////////    
	*dataReg1=_mm_setr_epi32(m1,M1,m2,M2);
    *dataReg2=_mm_setr_epi32(m3,M3,m4,M4);
    aux=_mm_min_epi32(*dataReg1,*dataReg2);
    *dataReg2=_mm_max_epi32(*dataReg1,*dataReg2);
    *dataReg1=aux;
////////////////////////////////////////////////////////////////////////
    m1=_mm_extract_epi32(*dataReg1,0);
    m2=_mm_extract_epi32(*dataReg1,1);
    m3=_mm_extract_epi32(*dataReg1,2);
    m4=_mm_extract_epi32(*dataReg1,3);

    M1=_mm_extract_epi32(*dataReg2,0);
    M2=_mm_extract_epi32(*dataReg2,1);
    M3=_mm_extract_epi32(*dataReg2,2);
    M4=_mm_extract_epi32(*dataReg2,3);
	//std::cout << "pasa 2" << std::endl;
    *dataReg1=_mm_setr_epi32(m1,M1,m2,M2);
    *dataReg2=_mm_setr_epi32(m3,M3,m4,M4);
    aux=_mm_min_epi32(*dataReg1,*dataReg2);
    *dataReg2=_mm_max_epi32(*dataReg1,*dataReg2);
    *dataReg1=aux;
	//std::cout << "pasa 3" << std::endl;
	//Código asociados a cada nivel del Bitonic Sorter
}
void print_m2(__m128i* Registros){
	std::cout << "-----------------Inicio de la m2---------------------" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[0],0) << "," << _mm_extract_epi32(Registros[0],1) << "," << _mm_extract_epi32(Registros[0],2) << "," << _mm_extract_epi32(Registros[0],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[1],0) << "," << _mm_extract_epi32(Registros[1],1) << "," << _mm_extract_epi32(Registros[1],2) << "," << _mm_extract_epi32(Registros[1],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[2],0) << "," << _mm_extract_epi32(Registros[2],1) << "," << _mm_extract_epi32(Registros[2],2) << "," << _mm_extract_epi32(Registros[2],3) << "]" << std::endl;
	std::cout << "[" << _mm_extract_epi32(Registros[3],0) << "," << _mm_extract_epi32(Registros[3],1) << "," << _mm_extract_epi32(Registros[3],2) << "," << _mm_extract_epi32(Registros[3],3) << "]" << std::endl;
	std::cout << "-----------------Termino de la m2---------------------" << std::endl;
}

//////////////////////////////////////////////////////////////////////
//  Bitonic Merge Network
//      **********************************************************
//      ***Importante: el paso de parámetros es por referencia.***
//      **********************************************************
//  Input:
//       __m128i*  dataReg  : Arreglo de 4 vectores ordenados
//                            individualmente
//
//  Output:
//      __m128i*  dataReg  : Arreglo de 4 vectores ordenados 
//                           globalmente
//
void BNM(__m128i*  dataReg){
	//Debe llamar a bitonicSorter() según el esquema
	//mostrado en clases
	
}

void uso(std::string pname)
{
	std::cerr << "Uso: " << pname << " --fname MATRIX_FILE" << std::endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{

	std::string fileName;
	
	//////////////////////////////////////////
	//  Read command-line parameters easy way
	if(argc != 3){
		uso(argv[0]);
	}
	std::string mystr;
	for (size_t i=0; i < argc; i++) {
		mystr=argv[i];
		if (mystr == "--fname") {
			fileName = argv[i+1];
		}
	}

	
	Timing timer0, timer1, timer3;
	////////////////////////////////////////////////////////////////
	// Transferir la matriz del archivo fileName a memoria principal
	timer0.start();
	MatrixToMem m1(fileName);
	timer0.stop();
	
	std::cout << "Time to transfer to main memory: " << timer0.elapsed() << std::endl;
	
	
	/*timer1.start();
	std::sort(m1._matrixInMemory, m1._matrixInMemory + m1._nfil);
	timer1.stop();
	
	std::cout << "Time to sort in main memory: " << timer1.elapsed() << std::endl;
	*/
	
	
	
	////////////////////////////////////////////////////////////////
	// Mostrar los N primeros elementos de la matriz desordenada.
	std::cout << "-----------Datos a ordenar---------" << std::endl;
	uint32_t N = 16;
	for(size_t i=0; i < N; i++){	
		std::cout << std::setw(8);	
		std::cout <<"dato " << i << " " << m1[i] << std::endl;	
	}
	
	
	__m128i  dataReg[4];
	dataReg[0] = _mm_setr_epi32(m1[0] , m1[1] , m1[2] , m1[3] );
	dataReg[1] = _mm_setr_epi32(m1[4] , m1[5] , m1[6] , m1[7] );
	dataReg[2] = _mm_setr_epi32(m1[8] , m1[9] , m1[10], m1[11]);
	dataReg[3] = _mm_setr_epi32(m1[12], m1[13], m1[14], m1[15]);
	
	//Ordenar los 4 datos de cada registro a través del Sorting Network
	sortNet(dataReg);
	print_m2(dataReg);
	transpose(dataReg);
	
	//Ordenar 8 datos en total de dos registros a través del Bitonic Sorter
	bitonicSorter(&dataReg[0], &dataReg[1]);
	bitonicSorter(&dataReg[2], &dataReg[3]);
	
	//Ordenar 16 datos a través de la Bitonic Merge Network
	BNM(dataReg);

	
	//Copiar el contenido de los registros vectoriales a memoria principal
	/*uint32_t* dest[16];
	dest[0]  = _mm_extract_epi32(dataReg[0], 0);
	dest[1]  = _mm_extract_epi32(dataReg[0], 1);
	dest[2]  = _mm_extract_epi32(dataReg[0], 2);
	dest[3]  = _mm_extract_epi32(dataReg[0], 3);
	
	dest[4]  = _mm_extract_epi32(dataReg[1], 0);
	dest[5]  = _mm_extract_epi32(dataReg[1], 1);
	dest[6]  = _mm_extract_epi32(dataReg[1], 2);
	dest[7]  = _mm_extract_epi32(dataReg[1], 3);
	
	dest[8]  = _mm_extract_epi32(dataReg[2], 0);
	dest[9]  = _mm_extract_epi32(dataReg[2], 1);
	dest[10] = _mm_extract_epi32(dataReg[2], 2);
	dest[11] = _mm_extract_epi32(dataReg[2], 3);
	
	dest[12] = _mm_extract_epi32(dataReg[3], 0);
	dest[13] = _mm_extract_epi32(dataReg[3], 1);
	dest[14] = _mm_extract_epi32(dataReg[3], 2);
	dest[15] = _mm_extract_epi32(dataReg[3], 3);
	
	
	
	std::cout << "-----------Datos Procesados-------" << std::endl;
	for(size_t i = 0; i < 16; i++){
		std::cout << std::setw(8);
		std::cout << dest[i] << std::endl;
	}*/


	
	/*
	std::cout << "-----------Shuffle example--------" << std::endl;
	std::cout << "Reg_i[0] original" << std::endl;
	for(size_t i = 0; i < 4; i++){
		std::cout << std::setw(8);
		std::cout << dest[i] << std::endl;
	}
	
	dataReg[0] = _mm_shuffle_epi32(dataReg[0], _MM_SHUFFLE(0, 1, 2, 3));
	
	dest[0]  = _mm_extract_epi32(dataReg[0], 0);
	dest[1]  = _mm_extract_epi32(dataReg[0], 1);
	dest[2]  = _mm_extract_epi32(dataReg[0], 2);
	dest[3]  = _mm_extract_epi32(dataReg[0], 3);
	
	std::cout << "Reg_i[0] with _MM_SHUFFLE(0, 1, 2, 3)" << std::endl;
	for(size_t i = 0; i < 4; i++){
		std::cout << std::setw(8);
		std::cout << dest[i] << std::endl;
	}
	*/
	

	return(EXIT_SUCCESS);
}


