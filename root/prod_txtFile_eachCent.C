void prod_txtFile_eachCent(){
    
    TFile *eventFile = new TFile("ampt_default_centrality.root","read");
    TTree *tree = (TTree*)eventFile->Get("hadronTree");
    	
    Float_t pt[244002];
    Float_t eta[244002];
    Float_t phi[244002];
    int nParticles=0;
    int nEvents;
    int cent;
    const int nCent = 10;
    int nEventCent[nCent]={0};
    int nEventCentIndex[nCent]={0};
    int centBoundary[nCent+1]={0,5,10,15,20,25,30,35,40,45,50};	

    tree->SetBranchAddress("nMultiplicityTree", &nParticles);
    tree->SetBranchAddress("centralityBinTree", &cent);
    tree->SetBranchAddress("ptTree", &pt);
    tree->SetBranchAddress("etaTree", &eta);
    tree->SetBranchAddress("phiTree", &phi);

    nEvents=tree->GetEntries();
    
    for(int i=0; i<nEvents; i++){
      if(i%200==0)  cout<<"Have run "<<i<<" of all the "<<nEvents<<" events; "<<endl;
      tree->GetEntry(i);
      for(int j=0; j<nCent; j++){
        if(cent>=centBoundary[j] && cent<centBoundary[j+1])
          nEventCent[j] = nEventCent[j]+1;
      }
    }
    int averageMult[10] = {0,0,0,0,0,0,0,0,0,0};
    ofstream outputFile[nCent];
    for(int j=0; j<nCent; j++){
      outputFile[j].open(Form("cent%d_default.txt",j));
      outputFile[j]<<nEventCent[j]<<endl;
    }
    for(int i=0; i<nEvents; i++){
      if(i%200==0)  cout<<"Have run "<<i<<" of all the "<<nEvents<<" events; "<<endl;
      tree->GetEntry(i);
      for(int j=0; j<nCent; j++){
        if(cent>=centBoundary[j] && cent<centBoundary[j+1]){
          averageMult[j] += nParticles;
          //outputFile[j] <<nParticles << endl;
	  outputFile[j]<<nEventCentIndex[j]<<"   " << nParticles<<"   "<< cent << endl;
          for(int k=0; k<nParticles; k++){
            //outputFile[j]<<eta[k]<<"   "<<phi[k]<<endl;
            outputFile[j]<<2*TMath::ATan(exp(-eta[k]))<<"   "<<phi[k]<<endl;
            //outputFile<< (2*TMath::ATan(exp(-eta[k]))) <<"   "<<phi[k]<<endl;
          //}

          nEventCentIndex[j] = nEventCentIndex[j]+1;

        }

      }
}
    }
    
    for(int i=0; i < 10; i++){
       averageMult[i] = averageMult[i]/nEventCent[i];
       //outputFile[i] << averageMult[i] << endl;
    }
}
