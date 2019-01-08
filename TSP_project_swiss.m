%% Data importation

lat = [47 24;
    47 23;
    47 20;
    47 28;
    47 34;
    46 57;
    46 48;
    46 12;
    47 02;
    46 51;
    47 22;
    47 3;
    47 0;
    46 57;
    46 53;
    47 42;
    47 1;
    47 13;
    47 25;
    47 33;
    46 12;
    46 52;
    46 14;
    46 31;
    47 10;
    47 22];

long = [8 3;
    9 16;
    9 25;
    7 44;
    7 36;
    7 27;
    7 9;
    6 9;
    9 4;
    9 32;
    7 21;
    8 18;
    6 56;
    8 21;
    8 14;
    8 38;
    8 39;
    7 32;
    9 22;
    8 54;
    9 1;
    8 38;
    7 22;
    6 38;
    8 31;
    8 33;];

lat_deg = [lat(:,1)+lat(:,2)/60];
long_deg = [long(:,1)+long(:,2)/60];
Cities = {'Aarau'; 'Herisau'; 'Appenzell'; 'Liestal'; 'Basel';
    'Bern'; 'Fribourg'; 'Genf'; 'Glarus'; 'Chur'; 'Delemont';
    'Lucerne'; 'Neuchatel'; 'Stans'; 'Sarnen'; 'Schaffausen';
    'Schwyz'; 'Solothurn'; 'StGallen'; 'Frauenfeld'; 'Bellinzona';
    'Altdorf'; 'Sion'; 'Lausanne'; 'Zug'; 'Zurich'};

gps = [lat_deg long_deg]; %in decimal degrees
gps_rad = gps.*pi/180; %in radians

%% GENETIC ALGORITHM
N = size(gps_rad,1);
FE = 200000;
popSize = 1000;
maxGen = FE/popSize;
pXO = 0.5;
pXOmax = 0.8; %Max XO rate for AXO
pMut = 0.001;
pMutmax = 0.02; %Max mutation rate for AMR

AMR = 0; %Adaptive Mutation Rate
AXO = 0; %Adaptive XO Rate (not used)
ROG = 0; %Random Offspring Generation
method = 'None'; %FVTLI1 or 2

%% Initialization

%Choose starting city for fun
StartCity = 'Genf';
[r, c] = find(strcmp(Cities, StartCity));
swap = Cities{1};
Cities{1} = StartCity;
Cities{r} = swap;
start = gps_rad(r,:);
gps_rad(r,:) = gps_rad(1,:);
gps_rad(1,:) = start;

%Planar approximation for GPS data
r = 6371;
x_pos = r*gps_rad(:,2)*cos(mean(gps_rad(:,1)));
y_pos = r*gps_rad(:,1);
x_pos = x_pos - x_pos(1);
y_pos = y_pos - y_pos(1);
xy_pos = [x_pos y_pos];

%Parent1 initialization
x = [ones(popSize,1) zeros(popSize, N-1) ones(popSize,1)];

for i=1:popSize
    x(i,2:end-1) = randperm(size(gps_rad,1)-1)+1; %Permute cities from 2-26
end

fit = nan(popSize,1);
fitN = fit;

for i=1:popSize
    fit(i) = EucliDist(x(i,:), x_pos, y_pos);
end

[fitEl, iEl] = min(fit);
xEl = x(iEl,:);

%% Run

for gen=2:maxGen
    
    %Ordered XO
    %Randomized subsetSize & randomized cut point for each offspring
    subsetSize=sum(rand(popSize,N-1) < pXO, 2);
    cut = floor(rand(popSize,1)*(N-1)+2);
    
    xN = [ones(popSize,1) zeros(popSize, N-1) ones(popSize,1)];
    parent2 = randperm(popSize);
    x2 = x(parent2,:);
    
    for i=1:popSize
        while cut(i) + subsetSize(i)-1 > N
            subsetSize(i) = subsetSize(i)-1;
        end
        subset = x(i,cut(i):(cut(i)+subsetSize(i)-1));
        xN(i,cut(i):(cut(i)+subsetSize(i)-1)) = subset;
    end
    
    %XO with parent 2
    %ROG
    for i=1:popSize
        if ROG == 1 & sum(x(i,:) == x2(i,:))==(N+1)
            x2(i,2:end-1) = randperm(N-1)+1;
            select = ~ismember(x2(i,:), xN(i,:));
            xN(i,xN(i,:) == 0) = x2(i,select);
        else
            select = ~ismember(x2(i,:), xN(i,:));
            xN(i,xN(i,:) == 0) = x2(i,select);
        end
    end
    
    %AMR
    if AMR == 1
        if gen > 0.8*maxGen
            pMut = pMut + (pMutmax-pMut)/(maxGen-0.8*maxGen);
        end
    end
    
    %AXO
    if AXO == 1
        if gen > 0.8*maxGen
            pXO = pXO + (pXOmax-pXO)/(maxGen-0.8*maxGen);
        end
    end
    
    %SWAP mutation
    SWAPmut = randperm(N-1,2)+1;
    mut=rand(popSize,N-1)<pMut;
    xN(logical(sum(mut,2)), SWAPmut) = xN(logical(sum(mut,2)), flip(SWAPmut));
    
    %Evaluate fitness
    for i=1:popSize
        fitN(i)=EucliDist(xN(i,:),x_pos, y_pos);
    end
    
    if method == "FVTLI1"
        %FVTLI1
        for i=1:popSize
            cut2 = floor(rand*(N-4)+2);
            subset2 = xN(i,cut2:cut2+3);
            x_temp = xN(i,:);
            x_temp(cut2:cut2+3) = [subset2(1) flip(subset2(2:3)) subset2(end)];
            if Euclidist(x_temp,x_pos,y_pos) < fitN(i)
                xN(i,:) = x_temp;
            end
        end
        
    else if method == "FVTLI2"
            %FVTLI2
            for i=1:popSize
                subsetSize = floor(rand*(N-2)+2);
                cut2 = N-subsetSize+1;
                subset2 = xN(i,cut2:cut2+subsetSize-1);
                x_temp = xN(i,:);
                x_temp(cut2:cut2+subsetSize-1) = flip(subset2);
                if EucliDist(x_temp,x_pos,y_pos) < fitN(i)
                    xN(i,:) = x_temp;
                end
            end
        end
    end
    
    %New population
    improved=fitN<fit;
    x(improved,:)=xN(improved,:);
    fit(improved)=fitN(improved);
    
    [fitNEl, iEl]=min(fitN);
    if fitNEl<fitEl
        xEl=xN(iEl,:);
        fitEl=fitNEl;
        disp([gen, fitEl]);
    end
    
    plot(x_pos(1),y_pos(1),'o','MarkerSize',12), hold on
    plot(x_pos,y_pos,'x')
    text(x_pos+1.5, y_pos, Cities,'Fontsize', 8),
    plot(x_pos(xEl),y_pos(xEl))
    plot(x_pos(xEl(1:2)),y_pos(xEl(1:2)),'b')
    title(fitEl);
    drawnow
    hold off
    
end
