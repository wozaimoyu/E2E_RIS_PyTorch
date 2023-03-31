clear;
close all;

BER_noRIS = load('Tradition_Ber_0.mat');
BER_noRIS = squeeze(mean(BER_noRIS.BER_noRIS,1));

BER128_trad = load('Tradition_Ber_128.mat');
BER128_trad = squeeze(mean(BER128_trad.BER,1));

BER256_trad = load('Tradition_Ber_256.mat');
BER256_trad = squeeze(mean(BER256_trad.BER,1));

BER128_e2e = load('E2E_Ber_128.mat');
BER128_e2e = squeeze(mean(BER128_e2e.Ber,1));

BER256_e2e = load('E2E_Ber_256.mat');
BER256_e2e = squeeze(mean(BER256_e2e.Ber,1));

for i=1:1
    close all;
    file  = sprintf("E2E_Ber_1024_0.mat");  % Change this
    %fig1 = sprintf("fig_a_%02d.png", i);
    %fig2 = sprintf("fig_b_%02d.png", i);
    fig1 = sprintf("mean_fig_a.gif");
    fig2 = sprintf("mean_fig_b.gif");
    fprintf(string(i) + "\n");
    
    if ~isfile(file)
        break
    end

    BER_my = load(file);
    BER_my = squeeze(mean(BER_my.Ber(1:i, :, :),1));
    
    figure(1)
    x = 0:10:100;
    k = 5;
    semilogy(x,BER_noRIS(:,k),'r-o','LineWidth',1.7,'color',[0.13 0.55 0.13]);
    hold on;
    semilogy(x,BER128_trad(:,k),'b-s','LineWidth',1.7); 
    semilogy(x,BER128_e2e(:,k),'r-v','LineWidth',1.7);
    semilogy(x,BER256_trad(:,k),'b--o','LineWidth',1.7);
    semilogy(x,BER256_e2e(:,k),'r--^','LineWidth',1.7);
    semilogy(x,BER_my(:,k),'k--^','LineWidth',1.7); 
    grid on
    axis([0 100 1e-6 1])
    %title(['RIS-aided communication system  SNR=-1dB'])
    xlabel('{\it L} (m)')
    ylabel('BER')
    legend({ ...
        'Rayleigh Without RIS', ...
        '128 - Alternating Scheme [6]', ...
        '128 - E2E Scheme', ...
        '256 - Alternating Scheme [6]', ...
        '256 - E2E Scheme', ...
        sprintf('My Scenario - %d', i) ...
        }, 'Location','southwest')
    ax = gca;
    exportgraphics(ax,fig1,'Resolution',300, 'Append', i ~= 1)
    %saveas(gcf,fig1)
    
    
    figure(2)
    x = -5:20;
    k = 3;
    semilogy(x,BER_noRIS(k,:),'r-o','LineWidth',1.7,'color',[0.13 0.55 0.13]);
    hold on;
    semilogy(x,BER128_trad(k,:),'b-s','LineWidth',1.7); 
    semilogy(x,BER128_e2e(k,:),'r-v','LineWidth',1.7);
    semilogy(x,BER256_trad(k,:),'b--o','LineWidth',1.7);
    semilogy(x,BER256_e2e(k,:),'r--^','LineWidth',1.7);
    semilogy(x,BER_my(k,:),'k--^','LineWidth',1.7); 
    grid on
    axis([-5 20 0.000001 1])
    %title('RIS-aided communication system L=20m')
    xlabel('SNR (dB)')
    ylabel('BER')
    
    legend({ ...
        'Rayleigh Without RIS', ...
        '128 - Alternating Scheme [6]', ...
        '128 - E2E Scheme', ...
        '256 - Alternating Scheme [6]', ...
        '256 - E2E Scheme', ...
        sprintf('My Scenario - %d', i) ...
        }, 'Location','southwest')
    
    ay = gca;
    exportgraphics(ay,fig2,'Resolution',300, 'Append', i ~= 1)
    %saveas(gcf,fig2)
end
fprintf("Done\n")