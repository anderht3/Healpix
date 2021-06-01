#include "TRandom3.h"
void production(){

    Float_t pt[20000];
    Float_t eta[20000];
    Float_t theta[20000];
    Float_t phi[20000];
    int nParticles=2000;
    int nEvents=100;


    ofstream outputFile;
    outputFile.open("eventFileFullSingle_iso.txt");
    
    TRandom3 *rnd;
    outputFile << nEvents<<endl;
    rnd = new TRandom3(0);
    for(int i=0; i<nEvents; i++){
      if(i%2==0)  cout<<"Have run "<<i<<" of all the "<<nEvents<<" events; "<<endl;
      outputFile << i << endl;
      for(int j=0; j<nParticles; j++){
        pt[j]=rnd->Uniform(0.3,10);
        phi[j]=rnd->Uniform(0.0,2*3.14159265358979);
        theta[j]=rnd->Uniform(0,180);
        //outputFile<<pt[j]<<"   "<<eta[j]<<"   "<<phi[j]<<endl;
        //outputFile<<pt[j]<<"   "<<2*TMath::ATan(exp(-eta[j]))<<"   "<<phi[j]<<endl;
        outputFile<<pt[j]<<"   "<<theta[j]*3.14159265358979/180<<"   "<<phi[j]<<endl;
      }
   
   }

}


