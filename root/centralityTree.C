#include <stdio.h>
#include <complex>
#include <iostream>
#include "TF1.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TRandom.h"
#include "TTree.h"
#include <TMath.h>
using namespace std; 
void centralityTree(){

     long nEvt;


     const int nCentralityBin = 100;
     Float_t boundariesBin[nCentralityBin+1] = {0, 1.5002, 2.1505, 2.6372, 3.0491, 3.3814, 3.6983, 3.9919, 4.2535, 4.496, 4.749, 4.9704, 5.1921, 5.4016, 5.6138, 5.8022, 5.9834, 6.1505, 6.3163, 6.4845, 6.6533, 6.8189, 6.9861, 7.1519, 7.3098, 7.4667, 7.6105, 7.769, 7.9307, 8.0702, 8.2119, 8.3431, 8.4815, 8.6212, 8.755, 8.8814, 8.9969, 9.1152, 9.2237, 9.3375, 9.459, 9.5749, 9.6968, 9.8161, 9.9267, 10.0363, 10.1394, 10.2449, 10.3468, 10.4685, 10.5684, 10.683, 10.7859, 10.8869, 10.9914, 11.0982, 11.2028, 11.3034, 11.4063, 11.5006, 11.5947, 11.6957, 11.7837, 11.8878, 11.9839, 12.072, 12.1647, 12.2453, 12.3384, 12.4219, 12.5084, 12.5885, 12.681, 12.7671, 12.8559, 12.9394, 13.0253, 13.1209, 13.2026, 13.2935, 13.3806, 13.4662, 13.5463, 13.6296, 13.7122, 13.7933, 13.876, 13.954, 14.0368, 14.1182, 14.1999, 14.2811, 14.36, 14.4318, 14.5139, 14.5972, 14.6734, 14.7472, 14.827, 14.9143, 15};

     Int_t centralityBinTree;
     Float_t b;

     TFile oldFile("./ampt_st_melting.root", "read");
     TTree *t1 = (TTree*)oldFile.Get("hadronTree");
     t1->SetBranchAddress("bTree", &b);
     nEvt=t1->GetEntries();

     TFile *newFile = new TFile("ampt_st_melting_centrality.root","recreate");
     cout<<"Making a clone of the tree ... ..."<<endl;
     TTree *newTree = t1->CloneTree();

     auto newBranch = newTree->Branch("centralityBinTree", &centralityBinTree, "centralityBinTree/I");
     for(long ne=0; ne<nEvt; ne++)
     {
       if(ne%2000==0)  cout<<"Have run "<<ne<<" of the total "<<nEvt<<" events; "<<endl;
       t1->GetEntry(ne);

       for(int i=0; i<nCentralityBin; i++){
         if(b>boundariesBin[i] && b<=boundariesBin[i+1]){
           centralityBinTree = i;
         }
       }

       newBranch->Fill();
       //newTree->Fill();

     } // end of the first event loop

  newTree->Write();
  newFile->Write();

       
}


