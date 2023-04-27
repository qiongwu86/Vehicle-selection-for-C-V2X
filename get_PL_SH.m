function [ PL , std_dev ] = get_PL_SH ( distance );


    % Parameters of the radio propagation model:
    fc = 5.91e9;                % Carrier frequency (Hz)
    hBS = 1.5;                  % Transmitter antenna height (m)
    hMS = 10;                  % Receiver antenna height (m)
    environmentHeight = 0;      % Average environmental height (m)
    distance = abs(distance);
    abc = 0;
    
    c = 3e8;
    Bo = c / fc;
    if distance >= 0
        m = sqrt(hMS.^2 + distance.^2);
        cos = distance / m;
        fd = (15 * cos) / Bo;
    else
        fd = 0;
    end
 
    dBP = 4 * (hBS-environmentHeight) * (hMS-environmentHeight) * (fc+fd) / c; % breakpoint distance

    % Avoid errors for very small distances:
    i = find(distance < 3);
    distance(i) = 3;

    % Calculate pathloss for distances lower than the breakpoint distance:
    i = find(distance < dBP);
    PL(i) = 22*log10(distance(i)) + 28 + 20*log10((fc+fd)/1e9);
    std_dev(i) = 3;    % Standard deviation

    % Calculate pathloss for distances higher than the breakpoint distance:
    i = find(distance >= dBP);
    PL(i) = 40*log10(distance(i)) + 7.8 - 18*log10(hBS-environmentHeight) - 18*log10(hMS-environmentHeight) + 2*log10((fc+fd)/1e9);
    std_dev(i) = 3;    % Standard deviation
    
    
    % Compares obtained pathloss with free-space pathloss:
    PLfree = 20*log10(distance) + 32.4 + 20*log10((fc+fd)*1e-9 / 5);
    if (PLfree > PL)
        i = find(PLfree > PL);
        PL(i) = PLfree(i);
        abc = abc + 1;
    end
    
end
