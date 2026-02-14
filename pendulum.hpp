using namespace std;
#include <tuple>

using namespace std;

class doublePendulum {
    double angle1;
    double angle2;
    double length1;
    double length2;
    double mass1;
    double mass2;
    double const g_constant;
    double deltaT;
    double t;

    doublePendulum(double angle1, double angle2, double length1, double length2
        double mass1, double mass2, double deltaT, double const g_constant,
        double t);
    
    tuple<double, double> calculatePosition(); // use the physics formula given the class atributes to output angles of pendulum 1 and 2
    

}

void  fillInCsvData();
/*
Angle1 range = (0,180), angle2 range = (0,360)
make 5 threads each covering 1/5 of possible angle combinations, like CUDA
for each thread:
    for loop iterating for angle1:
        for loop iterating for angle2:
            




*/