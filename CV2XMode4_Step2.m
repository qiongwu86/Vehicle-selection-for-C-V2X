function [ deltaCOL , CBR ] = CV2XMode4_Step2( beta , lambda , Pt , S , distance, Psen , step_dB , noise , coding , deltaPRO )


% Input parameters:
%   lambda: packet transmission frequency in Hz. .
%   Pt: transmission power in dBm. 
%   S: number of sub-channels. 4.
%   distance: distance between tramsmitter and receiver in meters. It can be a vector with multiple distances.
%   Psen: sensing threshold in dBm.
%   step_dB: discrete steps to compute the PDF of the SNR and SINR in dB.
%   noise: noise corresponding to the DATA field of each message. Assumes a noise figure of 9dB and 10MHz channel (background noise of -95dBm). The total number of RBs in 10MHz is 50.
%   coding: ID of the coding used to identify the BLER curve 
%   deltaPRO: probability of packet loss due to propagation effects for different Tx-Rx distances
% 
% Output metrics:
%   deltaCOL: probability of packet loss due to packet collisions for different Tx-Rx distances (between 0 and 1)
%   CBR: Channel Busy Ratio (between 0 and 1)
%
% The equations that are identified with a number between brackets in this script are the ones
% that also appear in the paper so that they can be easily identified. 

    N = S * 1000/lambda;  % Total number of resources in 1000ms. Equation (28)
    Nc = 0.2*N;           % Number of candidate resources (always 20% of N as specified in 3GPP standards)

    Na = 0; % Number of available resources
    d_aux = -500:500;   % Take into account interfering vehicles up to 1500 meters distance, since the interference generated at that thistance is almost null. 
    [PL, std_dev] = get_PL_SH(d_aux); % Calculates the pathloss and shadowing for a given set of interfering distances following the Winner+ B1 propagation model.
    
    while Na < Nc
        Psen = Psen + step_dB; % Increase threshold
        PSR = 0.5 * (1 + erf( (Pt - PL - Psen)./(std_dev*sqrt(2)) ) );  % PSR for all distances in d_aux. Equation (11)
    
        Spsr = beta * sum(PSR) ;  % Equation(29.1)
        Ne =  Spsr/2 + sum(max( 1 - (1:(Spsr/2))/(N-Spsr/2) , 0 )); % Number of excluded resources. Equation (29)
        Na = N - Ne; % Number of assignable resources.
        CBR = Ne / N; % Channel busy ratio. Equation (34)
    end
%     for d=1:length(distance)  
% %     Lint_max = round(400*beta)/beta;    % Distance to the farthest interfering vehicle. Up to 1000m distances considered to speed up the calculations. 
% %     distance_int_to_rx = [-Lint_max : 1/beta : Lint_max];           % Distances from all the interfering vehicles to the receiving vehicle.
%         distance_int_to_rx = distance;
%         distance_int_to_rx (d) = [];    % Remove from the list the position of the receiving vehicle. 
%     end
    
    R_PSR = xcorr(PSR);                 % Autocorrelation of the PSR function. Equation (30.2)
    R_PSR = R_PSR(2*max(d_aux)+1:end);  % Remove left part of the function            
    R0 = R_PSR(1); % Equation (30.1)
    
    Ce = ( R_PSR / R0 ) * (beta*Ne*R0/Spsr - Ne*Ne/N) + Ne*Ne/N; % Common excluded resources. Equation (30)  
    Ca = N - 2*Ne + Ce;  % Common available resources. Equation (27)
    Cc = Ca * (Nc/Na)^2; % Common candidate resources. Equation (26)      

    % Calculate probability of collision for Step 2 for each Tx-Rx distance:
    for d=1:length(distance)  
        
        [PL_E_R(d), std_dev_E_R(d)] = get_PL_SH(distance(d));  % Calculate the pathloss and shadowing for a given Tx-Rx distance following the Winner+ B1 propagation model.

        distance_int_to_rx = distance;
        distance_int_to_rx (d) = [];  
        distance_int_to_tx(d,:) = distance_int_to_rx - distance(d);  % Distances from all the interfering vehicles to the transmitting vehicle.

        % Calculate probability of collision for each interfering vehicle:
        for i = 1:length(distance_int_to_rx)    
            [PL_I_R , std_dev_I_R] = get_PL_SH(distance_int_to_rx(i));   % Pathloss and shadowing for interf and rx
            [PL_I_E , std_dev_I_E] = get_PL_SH(distance_int_to_tx(d,i)); % Pathloss and shadowing for interf and tx
            
            Pi_dB = Pt - PL_I_R; % Average received interference
            if deltaPRO(d) == 1       
                p_int(d,i) = 0;  % If the proability of packet loss due to propagation is 1
            else
                [SINR, PDF_SINR] = get_SINRdistribution( Pt-PL_E_R(d) , Pi_dB , std_dev_E_R(d) , std_dev_I_R , noise , Psen , step_dB); % PDF of the SINR experienced by the receiving vehicle
                p_SINR(d,i) = get_BLER ( SINR , PDF_SINR , coding , step_dB );     % Probability that the receiver receives a packet with error due to low SINR. Equation (17)
              
                p_int(d,i) = ( p_SINR(d,i) - deltaPRO(d) ) / ( 1 - deltaPRO(d) );  % Probability that the interference generated on the receiver provokes that the packet transmitted by vt cannot be correctly received at vr.  Equation (18)
            end

            tau = lambda;
            pSsensing_I_E = 0.5 * (1 + erf(( Pt - PL_I_E - Psen )/(2^0.5*std_dev_I_E)));  % PSR for interfering and transmitting vehicles
            p_s(i) = 1-(1-1/tau)*pSsensing_I_E; % Probability that transmitter and interferer do not take into account their respective transmissions before selecting a new resource. Equation (24)
            p_sim_2(i) = p_s(i) * Cc( round( abs(distance_int_to_tx(d,i))+ 1 )  ) / (Nc*Nc) ;  % Equation (23)

        end
        
        p_col = p_sim_2 .* p_int(d,:);      % Equation (15)
        deltaCOL(d) = 1 - prod(1 - p_col);  % Probability of packet loss due to collision when only Step 2 is executed. Equation (14)

    end
    %p_SINR
    %int=p_SINR;
    %dlmwrite('int_150_50_23_2.txt', int, 'delimiter', ',', '-append');
return


