// To force linking in all needed Intel routines
// See compileCX for details (linking against .a)
void Dummy(void)
{
	vsAdd();
	vsSub();
	vsDiv();
	vsSqr();
	vsMul();
	vsAbs();
	vsInv();

	vsSin();   
	vsCos();   
	vsSinCos();
	vsTan();   
	vsAsin();  
	vsAcos();  
	vsAtan();  
	vsAtan2();

	vsSinh(); 
	vsCosh(); 
	vsTanh(); 
	vsAsinh();
	vsAcosh();
	vsAtanh();

	vsPow();    
	vsPowx();   
	vsSqrt();   
	vsCbrt();   
	vsInvSqrt();
	vsInvCbrt();
	vsHypot();

	vsFloor();   
	vsCeil();    
	vsRound();   
	vsTrunc();   
	vsRint();    
	vsNearbyInt();
	vsModf();

	vsExp();  	     
	vsLn();   
	vsLog10();

	vsErf();   
	vsErfc();  
	vsErfInv();
}
