void figure4(){

  ifstream fileIn;
  fileIn.open("cl_full.txt");
  ifstream fileIn30;
  fileIn30.open("cl_eta30.txt");
  ifstream fileIn20;
  fileIn20.open("cl_eta20.txt");
  ifstream fileIn09;
  fileIn09.open("cl_eta09.txt");
 
  const int nL = 24;
  double cL[nL];
  double cL30[nL];
  double cL20[nL];
  double cL09[nL];
  double cLErr[nL];
  double cLErr30[nL];
  double cLErr20[nL];
  double cLErr09[nL];
  double xl[nL];
  double xl30[nL];
  double xl20[nL];
  double xl09[nL];

  for(int i=0; i< nL; i++){
    fileIn>>xl[i];
    fileIn>>cL[i];
    fileIn>>cLErr[i];
    fileIn30>>xl30[i];
    fileIn30>>cL30[i];
    fileIn30>>cLErr30[i];
    fileIn20>>xl20[i];
    fileIn20>>cL20[i];
    fileIn20>>cLErr20[i];
    fileIn09>>xl09[i];
    fileIn09>>cL09[i];
    fileIn09>>cLErr09[i];
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

 TH1D* hist = new TH1D("hist","",200,-0.3,18.7);
 hist->SetXTitle("#font[12]{l}");
 hist->SetYTitle("#LT#font[12]{C_{l}}#GT");
 hist->SetMinimum(0.0005);
 hist->SetMaximum(35.9);
 hist->GetXaxis()->CenterTitle(1);
 hist->GetYaxis()->CenterTitle(1);
 hist->GetYaxis()->SetTitleOffset(1.1);
 hist->GetXaxis()->SetTitleOffset(0.95);
 hist->GetXaxis()->SetTitleSize(0.066);
 hist->GetYaxis()->SetTitleSize(0.066);
 hist->GetXaxis()->SetLabelSize(0.045);
 hist->GetYaxis()->SetLabelSize(0.045);
 hist->Draw();

  TGraphErrors *gr2 = new TGraphErrors(19,xl,cL,0,cLErr);
  gr2->SetTitle("");
  gr2->SetMarkerStyle(25);
  gr2->SetMarkerSize(1.0);
  gr2->SetMarkerColor(1);
  gr2->SetLineWidth(2);
  gr2->SetLineColor(1);
  gr2->Draw("plsameez"); 
  TGraphErrors *gr3 = new TGraphErrors(19,xl30,cL30,0,cLErr30);
  gr3->SetTitle("");
  gr3->SetMarkerStyle(24);
  gr3->SetMarkerSize(1.0);
  gr3->SetMarkerColor(2);
  gr3->SetLineWidth(2);
  gr3->SetLineColor(2);
  gr3->Draw("plsameez");
  TGraphErrors *gr4 = new TGraphErrors(19,xl20,cL20,0,cLErr20);
  gr4->SetTitle("");
  gr4->SetMarkerStyle(21);
  gr4->SetMarkerSize(1.0);
  gr4->SetMarkerColor(4);
  gr4->SetLineWidth(2);
  gr4->SetLineColor(4);
  gr4->Draw("plsameez");
  TGraphErrors *gr5 = new TGraphErrors(19,xl09,cL09,0,cLErr09);
  gr5->SetTitle("");
  gr5->SetMarkerStyle(20);
  gr5->SetMarkerSize(1.0);
  gr5->SetMarkerColor(6);
  gr5->SetLineWidth(2);
  gr5->SetLineColor(6);
  gr5->Draw("plsameez");

    TLegend *leg = new TLegend(0.7,0.53,0.89,0.9);
    leg->SetFillColor(10);
    leg->SetBorderSize(0);
    leg->SetTextFont(42);
    leg->SetTextColor(1);
    leg->SetTextSize(0.05);
    //leg->SetLineStyle(0.06);
    leg->AddEntry(gr2,"full sky","pl");
    leg->AddEntry(gr3,"|#eta|<3.0","pl");
    leg->AddEntry(gr4,"|#eta|<2.0","pl");
    leg->AddEntry(gr5,"|#eta|<0.9","pl");
    leg->Draw();

    TLatex *tex2= new TLatex(5.1, 10.6,"Isotropic events");
    tex2->SetTextColor(1);
    tex2->SetTextSize(0.05);
    tex2->SetTextFont(42);
    tex2->Draw();



  c1->SaveAs("plot_figure4.png");
  c1->SaveAs("plot_figure4.pdf");

}

