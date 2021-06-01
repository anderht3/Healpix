void figure3(){

  ifstream fileIn;
  fileIn.open("ampt7_aps_wm.txt");
  ifstream fileInIso;
  fileInIso.open("iso7_single_aps_wm.txt");
 
  const int nL = 24;
  double cL[nL];
  double cLErr[nL];
  double xL[nL];
  double cLiso[nL];
  double cLisoErr[nL];
  double xLiso[nL];

  for(int i=0; i< nL; i++){
    fileIn>>xL[i];
    fileIn>>cL[i];
    fileIn>>cLErr[i];
    //cout<<xL[i]<<"   "<<cL[i]<<endl;
  }
  for(int i=0; i< nL; i++){
    fileInIso>>xLiso[i];
    fileInIso>>cLiso[i];
    fileInIso>>cLisoErr[i];
    //cout<<xLiso[i]<<"   "<<cLiso[i]<<endl;
  } 

  TCanvas *c1 = new TCanvas("c1","c1",1,1,650,550);
  c1->SetFillColor(10);
  c1->SetFrameFillColor(0);
  c1->SetFrameBorderSize(0);
  c1->SetFrameBorderMode(0);
  c1->SetLeftMargin(0.15);
  c1->SetBottomMargin(0.15);
  c1->SetTopMargin(0.05);
  c1->SetRightMargin(0.02);
  //c1->Divide(2,1,0,0);
  gStyle->SetOptStat(0);
  c1->SetTicks(-1);
  c1->SetLogy();  
  //c1->SetLogx();  

 TH1D* hist = new TH1D("hist","",200,-0.2,19.9);
 hist->SetXTitle("#font[12]{l}");
 hist->SetYTitle("#LT#font[12]{C_{l}}#GT");
 hist->SetMinimum(0.001);
 hist->SetMaximum(16.199);
 hist->GetXaxis()->CenterTitle(1);
 hist->GetYaxis()->CenterTitle(1);
 hist->GetYaxis()->SetTitleOffset(1.1);
 hist->GetXaxis()->SetTitleOffset(0.95);
 hist->GetXaxis()->SetTitleSize(0.066);
 hist->GetYaxis()->SetTitleSize(0.066);
 hist->GetXaxis()->SetLabelSize(0.045);
 hist->GetYaxis()->SetLabelSize(0.045);
 hist->Draw();

  TGraphErrors *gr2 = new TGraphErrors(nL,xL,cL,0,cLErr);
  gr2->SetTitle("");
  gr2->SetMarkerStyle(20);
  gr2->SetMarkerSize(1.0);
  gr2->SetMarkerColor(2);
  gr2->SetLineWidth(2);
  gr2->SetLineColor(2);
  gr2->Draw("plsameez"); 

  TGraphErrors *gr3 = new TGraphErrors(nL,xLiso,cLiso,0,cLisoErr);
  gr3->SetTitle("");
  gr3->SetMarkerStyle(25);
  gr3->SetMarkerSize(1.0);
  gr3->SetMarkerColor(4);
  gr3->SetLineWidth(2);
  gr3->SetLineColor(4);
  gr3->Draw("plsameez");


    TLegend *leg = new TLegend(0.65,0.77,0.8,0.9);
    leg->SetFillColor(10);
    leg->SetBorderSize(0);
    leg->SetTextFont(42);
    leg->SetTextColor(1);
    leg->SetTextSize(0.05);
    //leg->SetLineStyle(0.06);
    leg->AddEntry(gr2,"AMPT","pl");
    leg->AddEntry(gr3,"isotropic dist.","pl");
    leg->Draw();

    TLatex *tex2= new TLatex(12.1, 0.42,"35-40%, |#eta|<2.4");
    tex2->SetTextColor(1);
    tex2->SetTextSize(0.05);
    tex2->SetTextFont(42);
    tex2->Draw();

  c1->SaveAs("plot_figure3.png");
  c1->SaveAs("plot_figure3.pdf");

}

