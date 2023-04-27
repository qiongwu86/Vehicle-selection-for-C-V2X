function [ SINR , PDF_SINR ] =  get_SINRdistribution( Pr_dBm_avg , Pi_dBm_avg , std_dev_Pr , std_dev_Pi , noise_dBm , Psen , step_dB )


% Input parameters:
%   Pr_dBm_avg: average received power from the transmitting vehicle in dBm
%   Pi_dBm_avg: average received power from the interfering vehicle in dBm
%   std_dev_Pr: standard deviation of the shadowing experienced by the signal generated by the transmitting vehicle in dB
%   std_dev_Pi: standard deviation of the shadowing experienced by the signal generated by the interfering vehicle in dB
%   noise_dBm: background noise in dBm
%   sensingThreshold: sensing threshold in dBm
%   step_dB: discrete steps to compute the PDF of the SNR and SINR (dB)
%
% Output metrics:
%   SINR: Signal-to-Interference and Noise Ratio levels for different distances. It is a matrix where each row represent the SINR levels for a given Tx-Rx distance
%   PDF_SINR: Probability Density Function of the SINR levels for different distances. It is a matrix where each row represent the PDF for a given Tx-Rx distance
%
% The equations that are identified with a number between brackets in this script are the ones
% that also appear in the paper so that they can be easily identified. 
    abc = 0;
    x = -200:step_dB:200; % Wide range of values in dB to build the PDFs

    for i=1:length(Pr_dBm_avg)

        distrib_Pr = normpdf(x,Pr_dBm_avg(i),std_dev_Pr(i));    % PDF of the received signal
        ind = find( x < Psen);              
        distrib_Pr(ind) = 0;                                    % Remove values below the sensing threshold
        distrib_Pr = distrib_Pr / sum(distrib_Pr) / step_dB;    % Normalize so that the integral between -inf and +inf is equal to 1 

        if Pi_dBm_avg == -inf     
            % If there is no interference:
            distrib_Pi_noise = zeros(1,length(x));
            distrib_Pi_noise( round( (noise_dBm-x(1))/step_dB ) +1 ) = 1 / step_dB;
        else
            % If there is interference, compute the PDF of the SINR:
            aux = length( find(x <= noise_dBm) );   % Remove values below noise, since Pi+noise will never be lower than noise
            noise = 10^(noise_dBm/10);              % Noise power in linear units
            distrib_Pi_noise = 10.^(x(aux+1:end)/10)./((10.^(x(aux+1:end)/10)-noise)*std_dev_Pi*sqrt(2*pi)) .* exp( -(10*log10(10.^(x(aux+1:end)/10)-noise)-Pi_dBm_avg).^2/(2*std_dev_Pi^2) ); % PDF of interference + noise
            distrib_Pi_noise = [ zeros(1,aux) distrib_Pi_noise];                        % Null probability for values below the noise
            distrib_Pi_noise = distrib_Pi_noise / sum(distrib_Pi_noise) / step_dB;   	% Normalize so that the integral between -inf and +inf is equal to 1 
        end

        % Calculate PDF of SINR through the cross-correlation of the PDF of the received power and the PDF of the interference+noise:    
        PDF_SINR(i,:) = xcorr(distrib_Pr,distrib_Pi_noise);
        
        %if abc == 100
            %PDF_SINR
        %end
        PDF_SINR(i,:) = PDF_SINR(i,:) / sum(PDF_SINR(i,:)) / step_dB;    % Normalize so that the integral between -inf and +inf is equal to 1 
      
        abc = abc + 1;
        % Adapt the range of the x axes to the values provided by the xcorr function:
        SINR(i,:) = min(x)*2:step_dB:max(x)*2;

    end

end


