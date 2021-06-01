void figure8(){

  ifstream fileIn;
  fileIn.open("ampt0_aps.txt");
  ifstream fileIn_1;
  fileIn_1.open("ampt1_aps.txt");
  ifstream fileIn_2;
  fileIn_2.open("ampt2_aps.txt");
  ifstream fileIn_3;
  fileIn_3.open("ampt3_aps.txt");
  ifstream fileIn_4;
  fileIn_4.open("ampt4_aps.txt");
  ifstream fileIn_5;
  fileIn_5.open("ampt5_aps.txt");
  ifstream fileIn_6;
  fileIn_6.open("ampt6_aps.txt");
  ifstream fileIn_7;
  fileIn_7.open("ampt7_aps.txt");
  ifstream fileIn_8;
  fileIn_8.open("iso0_single_aps.txt");

  const int nL = 23;
  double cL[nL];
  double cLErr[nL];
  double xL[nL];
  double cL_1[nL];
  double cL_1Err[nL];
  double xL_1[nL];
  double cL_2[nL];
  double cL_2Err[nL];
  double xL_2[nL];
  double cL_3[nL];
  double cL_3Err[nL];
  double xL_3[nL];
  double cL_4[nL];
  double cL_4Err[nL];
  double xL_4[nL];
  double cL_5[nL];
  double cL_5Err[nL];
  double xL_5[nL];
  double cL_6[nL];
  double cL_6Err[nL];
  double xL_6[nL];
  double cL_7[nL];
  double cL_7Err[nL];
  double xL_7[nL];
  double iso[nL];
  double iso_Err[nL];
  double xL_8[nL];

  for(int i=0; i< nL; i++){
    fileIn>>xL[i];
    fileIn>>cL[i];
    fileIn>>cLErr[i];
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
    fileIn_4>>xL_4[i];
    fileIn_4>>cL_4[i];
    fileIn_4>>cL_4Err[i];
    fileIn_5>>xL_5[i];
    fileIn_5>>cL_5[i];
    fileIn_5>>cL_5Err[i];
    fileIn_6>>xL_6[i];
    fileIn_6>>cL_6[i];
    fileIn_6>>cL_6Err[i];
    fileIn_7>>xL_7[i];
    fileIn_7>>cL_7[i];
    fileIn_7>>cL_7Err[i];
    fileIn_8>>xL_8[i];
    fileIn_8>>iso[i];
    fileIn_8>>iso_Err[i];
    cout << iso[i] << " " << iso_Err[i] << endl;
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
 hist->SetMaximum(0.119);
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

  TGraphErrors *gr6 = new TGraphErrors(nL,xL_4,cL_4,0,cL_4Err);
  gr6->SetTitle("");
  gr6->SetMarkerStyle(27);
  gr6->SetMarkerSize(1.0);
  gr6->SetMarkerColor(2);
  gr6->SetLineWidth(2);
  gr6->SetLineColor(2);
  gr6->Draw("plsameez");

  TGraphErrors *gr7 = new TGraphErrors(nL,xL_5,cL_5,0,cL_5Err);
  gr7->SetTitle("");
  gr7->SetMarkerStyle(28);
  gr7->SetMarkerSize(1.0);
  gr7->SetMarkerColor(1);
  gr7->SetLineWidth(2);
  gr7->SetLineColor(1);
  gr7->Draw("plsameez");

  TGraphErrors *gr8 = new TGraphErrors(nL,xL_6,cL_6,0,cL_6Err);
  gr8->SetTitle("");
  gr8->SetMarkerStyle(33);
  gr8->SetMarkerSize(1.0);
  gr8->SetMarkerColor(4);
  gr8->SetLineWidth(2);
  gr8->SetLineColor(4);
  gr8->Draw("plsameez");

  TGraphErrors *gr9 = new TGraphErrors(nL,xL_7,cL_7,0,cL_7Err);
  gr9->SetTitle("");
  gr9->SetMarkerStyle(34);
  gr9->SetMarkerSize(1.0);
  gr9->SetMarkerColor(8);
  gr9->SetLineWidth(2);
  gr9->SetLineColor(8);
  gr9->Draw("plsameez");


  TGraphErrors *gr10 = new TGraphErrors(nL,xL_8,iso,0,iso_Err);
  gr10->SetTitle("");
  gr10->SetMarkerStyle(34);
  gr10->SetMarkerSize(1.0);
  gr10->SetMarkerColor(1);
  gr10->SetLineWidth(2);
  gr10->SetLineColor(1);
  gr10->Draw("plsameez");


    TLegend *leg = new TLegend(0.58,0.72,0.71,0.91);
    leg->SetFillColor(10);
    leg->SetBorderSize(0);
    leg->SetTextFont(42);
    leg->SetTextColor(1);
    leg->SetTextSize(0.05);
    //leg->SetLineStyle(0.06);
    leg->AddEntry(gr2,"0-5%","pl");
    leg->AddEntry(gr3,"5-10%","pl");
    leg->AddEntry(gr4,"10-15%","pl");
    leg->AddEntry(gr5,"15-20%","pl");
    leg->Draw();

    TLegend *leg2 = new TLegend(0.78,0.72,0.91,0.91);
    leg2->SetFillColor(10);
    leg2->SetBorderSize(0);
    leg2->SetTextFont(42);
    leg2->SetTextColor(1);
    leg2->SetTextSize(0.05);
    //leg2->SetLineStyle(0.06);
    leg2->AddEntry(gr6,"20-25%","pl");
    leg2->AddEntry(gr7,"25-30%","pl");
    leg2->AddEntry(gr8,"30-35%","pl");
    leg2->AddEntry(gr9,"35-40%","pl");
    leg2->AddEntry(gr10,"iso 0-5%", "pl");
    leg2->Draw();

    TLatex *tex2= new TLatex(3.1, 0.065,"|#eta|<2.4");
    tex2->SetTextColor(1);
    tex2->SetTextSize(0.05);
    tex2->SetTextFont(42);
    tex2->Draw();

  c1->SaveAs("plot_figure8.png");
  c1->SaveAs("plot_figure8.pdf");

}

