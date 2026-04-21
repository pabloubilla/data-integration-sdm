library(RISDM)
library(terra)
library(ggplot2)

covars <- rast(system.file('extdata', 'ACT_DemoData.grd', package = 'RISDM'))
names(covars) <- c('lACC', 'SMRZ', 'TEMP')

dat <- simulateData.isdm(
  # distCoefs=covars[[c('SMRZ', 'TEMP')]],
  # biasCoefs=covars[['lACC']],
  # control = list(doPlot=FALSE)
)

plot(dat$covarBrick$var1,add=T)
meshy <- makeMesh(dat$covarBrick$var1, max.n = c(1000, 500),
                  dep.range = 0.5, offset = 10, expans.mult = 7.5, doPlot = TRUE)



fm <- isdm(observationList = list(PAdat = pa_train_df),
           covars = dat$covarBrick,
           mesh = meshy,
           responseNames = c(PA = "PA"),
           sampleAreaNames = c(PO = NULL, PA = "transectArea"),
           distributionFormula = ~0 + var1,
           # biasFormula = ~1 ,
           artefactFormulas = list(PA = ~1),
           control = list(prior.range = c(0.5, 0.1),
                          prior.space.sigma = c(2, 0.1),
                          coord.names = c("x", "y")))


summary(fm)


library(inlabru)

meshSF=as_sf

ggplot()+gg(meshy)
+ gg(dat$covarBrick$var1)

