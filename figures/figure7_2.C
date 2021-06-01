void figure7_2(){

  ifstream fileIn_1;
  fileIn_1.open("iso9_single_aps_wm.txt");
  ifstream fileIn_2;
  fileIn_2.open("iso9_gauss_aps_wm.txt");
  ifstream fileIn_3;
  fileIn_3.open("iso9_exact_aps_wm.txt");

  const int nL = 23;
  double cL_1[nL];
  double cL_1Err[nL];
  double xL_1[nL];
  double cL_2[nL];
  double cL_2Err[nL];
  double xL_2[nL];
  double cL_3[nL];
  double cL_3Err[nL];
  double xL_3[nL];

  for(int i=0; i< nL; i++){
    //cout<<xL[i]<<"   "<<cL[i]<<endl;
    fileIn_1>>xL_1[i];
    fileIn_1>>cL_1[i];
    fileIn_1>>cL_1Err[i];
    fileIn_2>>xL_2[i];
    fileIn_2>>cL_2[i];
    fileIn_2>>cL_2Err[i];
    fileIn_3>>xL_3[i];
    fileIn_3>>cL_3[i];
    fileIn_3>>cL_3Err[i];
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

 TH1D* hist = new TH1D("hist","",200,0.001,12.9);
 hist->SetXTitle("#font[12]{l}");
 hist->SetYTitle("#LT#font[12]{C_{l}^{m#neq0}}#GT");
 hist->SetMinimum(0.0007);
 hist->SetMaximum(14);
 hist->GetXaxis()->CenterTitle(1);
 hist->GetYaxis()->CenterTitle(1);
 hist->GetYaxis()->SetTitleOffset(1.1);
 hist->GetXaxis()->SetTitleOffset(0.95);
 hist->GetXaxis()->SetTitleSize(0.066);
 hist->GetYaxis()->SetTitleSize(0.066);
 hist->GetXaxis()->SetLabelSize(0.045);
 hist->GetYaxis()->SetLabelSize(0.045);
 hist->Draw();

  TGraphErrors *gr3 = new TGraphErrors(nL,xL_1,cL_1,0,cL_1Err);
  gr3->SetTitle("");
  gr3->SetMarkerStyle(21);
  gr3->SetMarkerSize(1.0);
  gr3->SetMarkerColor(1);
  gr3->SetLineWidth(2);
  gr3->SetLineColor(1);
  gr3->Draw("plsameez");

  TGraphErrors *gr4 = new TGraphErrors(nL,xL_2,cL_2,0,cL_2Err);
  gr4->SetTitle("");
  gr4->SetMarkerStyle(24);
  gr4->SetMarkerSize(1.0);
  gr4->SetMarkerColor(4);
  gr4->SetLineWidth(2);
  gr4->SetLineColor(4);
  gr4->Draw("plsameez");

  TGraphErrors *gr5 = new TGraphErrors(nL,xL_3,cL_3,0,cL_3Err);
  gr5->SetTitle("");
  gr5->SetMarkerStyle(25);
  gr5->SetMarkerSize(1.0);
  gr5->SetMarkerColor(8);
  gr5->SetLineWidth(2);
  gr5->SetLineColor(8);
  gr5->Draw("plsameez");


    TLegend *leg = new TLegend(0.58,0.72,0.71,0.91);
    leg->SetFillColor(10);
    leg->SetBorderSize(0);
    leg->SetTextFont(42);
    leg->SetTextColor(1);
    leg->SetTextSize(0.03);
    //leg->SetLineStyle(0.06);
    leg->AddEntry(gr3,"Isotropic Single Value 45-50%","pl");
    leg->AddEntry(gr4,"Isotropic Gaussian Values 45-50%","pl");
    leg->AddEntry(gr5,"Isotropic Exact Values 45-50%","pl");
    leg->Draw();


    TLatex *tex2= new TLatex(11, 0.068,"|#eta|<2.4");
    tex2->SetTextColor(1);
    tex2->SetTextSize(0.03);
    tex2->SetTextFont(42);
    tex2->Draw();

  c1->SaveAs("plot_figure7_2.png");
  c1->SaveAs("plot_figure7_2.pdf");

}

