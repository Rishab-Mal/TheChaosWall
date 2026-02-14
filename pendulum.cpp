tuple<double, double> doublePendulum::calculatePosition(){
    // input: use class attributes for formula
    // output: the 2 angles of the pendulums and
    // change state of attributes

    // Function Body: Whatever the physics logic is
}

doublePendulum::doublePendulum(double angle1, double angle2, double length1, double length2
    double mass1, double mass2, double deltaT, double const g_constant,
    double t)
{
    this->angle1 = angle1;
    this->angle2 = angle2;
    this->length1 = length1;
    this->length2 = length2;
    this-> mass1 = mass1;
    this->mass2 = mass2;
    this->deltaT = deltaT;
    this->g_constant = g_constant;
    this->t = t;
}