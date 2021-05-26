#include <global.hh>
#include <algorithm>
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
	bitonicSorter(&dataReg[0],&dataReg[1]);
	bitonicSorter(&dataReg[2],&dataReg[3]);
	bitonicSorter(&dataReg[1],&dataReg[2]);
	bitonicSorter(&dataReg[0],&dataReg[1]);
	bitonicSorter(&dataReg[2],&dataReg[3]);	
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

	
	Timing timer0, timer1, timer2, timer3;
	////////////////////////////////////////////////////////////////
	// Transferir la matriz del archivo fileName a memoria principal
	timer0.start();
	MatrixToMem m1(fileName);
	timer0.stop();
	std::cout << "********************************************"<< std::endl;
	std::cout << "Time to transfer to main memory: " << timer0.elapsed() << std::endl;
	std::cout << "--------------------------------------------"<< std::endl;
	
	timer1.start();
	std::sort(m1._matrixInMemory, m1._matrixInMemory + m1._nfil);
	timer1.stop();
	
	std::cout << "Time to sort in main memory: " << timer1.elapsed() << std::endl;
	std::cout << "********************************************"<< std::endl;
	
	
	
	////////////////////////////////////////////////////////////////
	// Mostrar los N primeros elementos de la matriz desordenada.
	MatrixToMem m2(fileName);
	std::cout << "-----------Datos a ordenar---------" << std::endl;
	uint32_t N = 16;
	for(size_t i=0; i < N; i++){	
		std::cout << std::setw(8);	
		std::cout <<"dato " << i << " " << m2[i] << std::endl;	
	}
	
	
	__m128i  dataReg[4];
	timer2.start();
	for(size_t i=0; i<m2._nfil; i+=16){
		dataReg[0] = _mm_setr_epi32(m2._matrixInMemory[i],m2._matrixInMemory[i+1],m2._matrixInMemory[i+2],m2._matrixInMemory[i+3]);
		dataReg[1] = _mm_setr_epi32(m2._matrixInMemory[i+4],m2._matrixInMemory[i+5],m2._matrixInMemory[i+6],m2._matrixInMemory[i+7]);
		dataReg[2] = _mm_setr_epi32(m2._matrixInMemory[i+8],m2._matrixInMemory[i+9],m2._matrixInMemory[i+10],m2._matrixInMemory[i+11]);
		dataReg[3] = _mm_setr_epi32(m2._matrixInMemory[i+12],m2._matrixInMemory[i+13],m2._matrixInMemory[i+14],m2._matrixInMemory[i+15]);
		//Ordenar los 4 datos de cada registro a través del Sorting Network
		sortNet(dataReg);
		transpose(dataReg);
		//Ordenar 8 datos en total de dos registros a través del Bitonic Sorter
		bitonicSorter(&dataReg[0], &dataReg[1]);
		bitonicSorter(&dataReg[2], &dataReg[3]);
		//Ordenar 16 datos a través de la Bitonic Merge Network
		BNM(dataReg);
		transpose(dataReg);
		m2._matrixInMemory[i] = _mm_extract_epi32(dataReg[0],0);
		m2._matrixInMemory[i+1] = _mm_extract_epi32(dataReg[0],1);
		m2._matrixInMemory[i+2] = _mm_extract_epi32(dataReg[0],2);
		m2._matrixInMemory[i+3] = _mm_extract_epi32(dataReg[0],3);
		m2._matrixInMemory[i+4] = _mm_extract_epi32(dataReg[1],0);
		m2._matrixInMemory[i+5] = _mm_extract_epi32(dataReg[1],1);
		m2._matrixInMemory[i+6] = _mm_extract_epi32(dataReg[1],2);
		m2._matrixInMemory[i+7] = _mm_extract_epi32(dataReg[1],3);
		m2._matrixInMemory[i+8] = _mm_extract_epi32(dataReg[2],0);
		m2._matrixInMemory[i+9] = _mm_extract_epi32(dataReg[2],1);
		m2._matrixInMemory[i+10] = _mm_extract_epi32(dataReg[2],2);
		m2._matrixInMemory[i+11] = _mm_extract_epi32(dataReg[2],3);
		m2._matrixInMemory[i+12] = _mm_extract_epi32(dataReg[3],0);
		m2._matrixInMemory[i+13] = _mm_extract_epi32(dataReg[3],1);
		m2._matrixInMemory[i+14] = _mm_extract_epi32(dataReg[3],2);
		m2._matrixInMemory[i+15] = _mm_extract_epi32(dataReg[3],3);
		break;
	}
	timer2.stop();
////////////////////////////////////////////////////////////////////////////////
	timer3.start();
	std::sort(m2._matrixInMemory, m2._matrixInMemory + m2._nfil);
	timer3.stop();
	
	std::cout << "********************************************"<< std::endl;
	std::cout << "Tiempo de ordenamiento vectorial "<< timer2.elapsed() <<std::endl;
	std::cout << "--------------------------------------------"<< std::endl;
	std::cout << "Tiempo de ordenamiento con sort "<< timer3.elapsed() <<std::endl;
	std::cout << "********************************************"<< std::endl;

	return(EXIT_SUCCESS);
}


