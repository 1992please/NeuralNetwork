#pragma once

#include "Assert.h"

template<typename T>
struct Mat
{
private:
	T **M;
	size_t nRows, nCols;
public:
	Mat<T>()
	{
		nRows = 0;
		nCols = 0;
		M = nullptr;
	}

	Mat<T>(const size_t _nRows, const size_t _nCols)
	{
		nRows = _nRows;
		nCols = _nCols;

		M = new T*[nRows];
		for (size_t i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
		}
	}

	Mat<T>(const size_t _nRows, const size_t _nCols, const T& _init)
	{
		nRows = _nRows;
		nCols = _nCols;

		M = new T*[nRows];
		for (size_t i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
			for (size_t j = 0; j < nCols; j++)
				M[i][j] = _init;
		}
	}

	Mat<T>(size_t Row, size_t Col, T* arr)
	{
		nRows = Row;
		nCols = Col;

		M = new T*[nRows];
		for (size_t i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
			for (size_t j = 0; j < nCols; j++)
				M[i][j] = *((arr + i * Col) + j);
		}
	}

	Mat<T>(const Mat<T>& nM)
	{
		nRows = nM.nRows;
		nCols = nM.nCols;

		M = new T*[nRows];
		for (size_t i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
			for (size_t j = 0; j < nCols; j++)
			{
				M[i][j] = nM.M[i][j];
			}
		}
	}

	Mat<T>(Mat<T>&& nM)
	{
		nRows = nM.nRows;
		nCols = nM.nCols;
		M = nM.M;

		nM.M = nullptr;
		nM.nRows = 0;
		nM.nCols = 0;
	}

	void ResizeClear(size_t _nRows, size_t _nCols)
	{
		Clear();
		nRows = _nRows;
		nCols = _nCols;

		M = new T*[nRows];
		for (size_t i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
		}
	}

	inline size_t RowsCount() const
	{
		return nRows;
	}

	inline size_t ColsCount() const
	{
		return nCols;
	}

	~Mat()
	{
		Clear();
	}

	void Clear()
	{
		nCols = 0;
		nRows = 0;

		if (!M)
			return;

		for (size_t i = 0; i < nRows; i++)
			delete[] M[i];

		delete[] M;
	}

	void Fill(const T& value)
	{
		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				M[i][j] = value;
	}

	inline Mat<T> Transposed()
	{
		Mat<T> mat(nCols, nRows);
		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				mat[j][i] = M[i][j];

		return mat;
	}

	inline Mat<T> Op(T(*Func)(double))
	{
		Mat<T> mat(nRows, nCols);
		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				mat[i][j] = Func(M[i][j]);
		return mat;
	}

	inline void operator=(const Mat& nM)
	{
		Clear();
		nRows = nM.nRows;
		nCols = nM.nCols;

		M = new T*[nRows];
		for (size_t i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
			for (size_t j = 0; j < nCols; j++)
			{
				M[i][j] = nM.M[i][j];
			}
		}
	}

	inline void operator=(Mat&& nM)
	{
		Clear();
		nRows = nM.nRows;
		nCols = nM.nCols;
		M = nM.M;

		nM.M = nullptr;
		nM.nRows = 0;
		nM.nCols = 0;
	}

	inline Mat<T> operator+(const Mat& rhs)
	{
		ASSERT(nRows == rhs.nRows && nCols == rhs.nCols);

		Mat<T> out(nRows, nCols);

		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				out[i][j] = M[i][j] + rhs[i][j];

		return out;
	}

	inline Mat<T> operator-(const Mat& rhs)
	{
		ASSERT(nRows == rhs.nRows && nCols == rhs.nCols);

		Mat<T> out(nRows, nCols);

		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				out[i][j] = M[i][j] - rhs[i][j];

		return out;
	}

	// we need to find a way to make this faster
	inline Mat<T> operator*(const Mat& rhs)
	{
		ASSERT(nCols == rhs.nRows);

		Mat<T> out(nRows, rhs.nCols, (T)0);

		for (size_t i = 0; i < out.nRows; i++)
			for (size_t j = 0; j < out.nCols; j++)
				for (size_t k = 0; k < nCols; k++)
					out[i][j] += M[i][k] * rhs[k][j];

		return out;
	}



	inline Mat<T> operator*(const T s)
	{
		Mat<T> out(nRows, nCols);
		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				out[i][j] = M[i][j] * s;

		return out;
	}

	inline Mat<T> operator+(const T s)
	{
		Mat<T> out(nRows, nCols);
		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				out[i][j] = M[i][j] + s;

		return out;
	}

	inline Mat<T> operator-(const T s)
	{
		Mat<T> out(nRows, nCols);
		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				out[i][j] = M[i][j] - s;

		return out;
	}

	inline Mat<T> operator-()
	{
		Mat<T> out(nRows, nCols);

		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				out[i][j] = -M[i][j];

		return out;
	}

	inline Mat<T> ComWiseMul(const Mat& rhs)
	{
		ASSERT(nRows == rhs.nRows && nCols == rhs.nCols);
		Mat<T> out(nRows, nCols);

		for (size_t i = 0; i < out.nRows; i++)
			for (size_t j = 0; j < out.nCols; j++)
				out[i][j] = M[i][j] * rhs[i][j];

		return out;
	}

	inline Mat<T> ComWiseSquared()
	{
		Mat<T> out(nRows, nCols);

		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				out[i][j] = M[i][j] * M[i][j];

		return out;
	}

	inline double Sum()
	{
		double out = 0;

		for (size_t i = 0; i < nRows; i++)
			for (size_t j = 0; j < nCols; j++)
				out += M[i][j];

		return out;
	}

	inline Mat<T> KroneckerMul(const Mat& rhs)
	{

		const size_t p = rhs.nRows;
		const size_t q = rhs.nCols;

		Mat<T> out(nRows * p, nCols * q);

		for (size_t r = 0; r < nRows; r++)
			for (size_t s = 0; s < nCols; s++)
				for (size_t v = 0; v < rhs.nRows; v++)
					for (size_t w = 0; w < rhs.nCols; w++)
						out[p * r + v][q * s + w] = M[r][s] * rhs[v][w];

		return out;
	}

	Mat<T> GetRow(size_t Index)
	{
		ASSERT(Index < nRows)
			Mat<T> out(1, nCols);
		for (size_t i = 0; i < nCols; i++)
		{
			out[0][i] = M[Index][i];
		}
		return out;
	}

	Mat<T> GetRows(size_t from, size_t to = 0)
	{
		to = to ? to : nRows;
		ASSERT(from < nRows && to <= nRows);
		Mat<T> out(to - from, nCols);
		for (size_t i = from; i < to; i++)
			for (size_t j = 0; j < nCols; j++)
				out[i - from][j] = M[i][j];
		return out;
	}

	Mat<T> GetCols(size_t from, size_t to = 0)
	{
		to = to ? to : nCols;
		ASSERT(from < nCols && to <= nCols);
		Mat<T> out(nRows, to - from);
		for (size_t i = 0; i < nRows; i++)
			for (size_t j = from; j < to; j++)
				out[i][j - from] = M[i][j];
		return out;
	}

	Mat<T> HConCat(const Mat& rhs)
	{
		ASSERT(nRows == rhs.nRows);
		Mat<T> out(nRows, nCols + rhs.nCols);

		for (size_t i = 0; i < out.nRows; i++)
			for (size_t j = 0; j < out.nCols; j++)
			{
				if (j < nCols)
					out[i][j] = M[i][j];
				else
					out[i][j] = rhs[i][j - nCols];
			}
		return out;
	}

	Mat<T> VConCat(const Mat& rhs)
	{
		ASSERT(nCols == rhs.nCols);
		Mat<T> out(nRows + rhs.nRows, nCols);

		for (size_t i = 0; i < out.nRows; i++)
			for (size_t j = 0; j < out.nCols; j++)
			{
				if (i < nRows)
					out[i][j] = M[i][j];
				else
					out[i][j] = rhs[i - nRows][j];
			}
		return out;
	}

	void Show(char * header = nullptr)
	{
		if (!header)
		{
			header = "MTemp";
		}

		printf("%s =\n", header);
		for (size_t i = 0; i < nRows; i++)
		{
			for (size_t j = 0; j < nCols; j++)
				printf("%.2f\t", M[i][j]);
			printf("\n");
		}
	}

	inline T* operator[](size_t row) const { return M[row]; }
};

template<typename T>
inline Mat<T> operator*(T const& s, Mat<T> rhs)
{
	return rhs * s;
}

template<typename T>
inline Mat<T> operator-(T const& s, Mat<T> rhs)
{
	Mat<T> out(rhs.RowsCount(), rhs.ColsCount());
	for (size_t i = 0; i < rhs.RowsCount(); i++)
		for (size_t j = 0; j < rhs.ColsCount(); j++)
			out[i][j] = s - rhs[i][j];

	return out;
}